# -*- coding: utf-8 -*-

import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.rankings import get_score_rankings, rank_query, invert_rankings  # , get_candidate_score_rankings
import multileaving.SampleBasedProbabilisticMultileave as pm
import multileaving.candidatepreselection as cps
from algorithms.probmeanbandit import ProbMeanBandit
import numpy as np


class EmbeddingBandit(ProbMeanBandit):

    def __init__(self, embedding_arguments, mgd_arguments, n_raw_features, k):

        # The ProbMeanBandit only sees the embedding features, thus feature_count = n_embedding_features
        self.n_raw_features = n_raw_features
        self.drop_prob = embedding_arguments.drop_probability
        self.drop_decay = embedding_arguments.drop_decay
        self.includes_linear_model = embedding_arguments.add_linear_model
        assert self.drop_decay < 1.
        if embedding_arguments.n_embedding_features > 0:
            self.n_embedding_features = embedding_arguments.n_embedding_features
        else:
            self.n_embedding_features = n_raw_features
        self.permanent_drop = embedding_arguments.permanent_drop
        self.min_updates_for_drop = embedding_arguments.min_updates_for_drop
        self.enable_drop = embedding_arguments.enable_drop
        self.min_embedding_features = embedding_arguments.min_embedding_features

        assert self.enable_drop or self.drop_prob == 0

        self.store_train_embeddings = self.drop_prob == 0
        self.new_features = np.ones(self.n_embedding_features)
        # method keeps track of embeddings so that we only change parts of embedding that have changed
        self.test_embeddings = None
        self.train_embeddings = None
        self.update_embedding = True

        self.setup_sampler()

        if self.includes_linear_model:
            ProbMeanBandit.__init__(self, mgd_arguments, self.n_embedding_features
                                    + n_raw_features, k, exploration=True)
        else:
            ProbMeanBandit.__init__(self, mgd_arguments, self.n_embedding_features, k,
                                    exploration=True)

        if self.enable_drop:
            self.add_message('N_DROP', 0)

    def clean(self):
        # explicitely clean out the embeddings
        del self.test_embeddings
        del self.train_embeddings
        del self.update_embedding
        gc.collect()
        ProbMeanBandit.clean(self)

    def setup_sampler(self):
        """
        Use this function if sampler needs to be setup before use.
        """
        pass

    def create_candidates(self):
        """
        Create candidates invokes the sample_candidates function and adds the current_best.
        """
        self.weights = np.zeros((self.feature_count, self.n_cand + 1))
        self.weights[:, :-1] = self.sample_candidates()
        self.weights[:, -1:] = self.current_best

    def sample_candidates(self):
        raise NotImplementedError('No Candidate sampling method implemented.')

    def generate_drop_weights(self, n_generate=None):
        assert not self.includes_linear_model or self.drop_prob == 0
        if n_generate is None:
            n_generate = self.n_generate
        vectors = np.random.randn(self.feature_count, n_generate)
        self.drop_mask = np.zeros((self.feature_count, n_generate))
        if self.feature_count > self.min_embedding_features:
            vectors, self.drop_mask = self.generate_drop_weights_rec(n_generate, vectors,
                    self.drop_mask, self.drop_prob)
        else:
            vectors, self.drop_mask = self.generate_drop_weights_rec(n_generate, vectors,
                    self.drop_mask, 0)
        weights = self.current_best[:, 0][:, None] + vectors * self.unit
        return weights

    def generate_drop_weights_rec(self, n_generate, vectors, full_drop_mask, drop_prob):
        """
        Vectors should be initialized with random.randn
        """
        if drop_prob == 0:
            vectors /= np.sum(vectors ** 2, axis=0) ** (1. / 2)
            return vectors, full_drop_mask
        else:
            drop_mask = np.random.rand(self.feature_count, n_generate) < drop_prob
            keep_mask = np.logical_not(drop_mask)

            if not np.any(keep_mask):
                return self.generate_drop_weights_rec(n_generate, vectors, full_drop_mask,
                        drop_prob * self.drop_decay)

            drop_v = np.tile(self.current_best[:, 0], (n_generate, 1)).T[drop_mask]
            vectors[drop_mask] = -drop_v / self.alpha / self.unit

            a = np.sum((vectors * drop_mask) ** 2, axis=0)
            b = np.sum((vectors * keep_mask) ** 2, axis=0)

            safe_bool = np.logical_and(b > 0, a <= 1)
            safe_ind = np.arange(n_generate)[safe_bool]
            squared = (1. - a[safe_ind]) / b[safe_ind]
            unsafe_ind = np.arange(n_generate)[np.logical_not(safe_bool)]
            n_unsafe = unsafe_ind.shape[0]

            # resample the setted weights
            vectors[:, unsafe_ind] = np.random.randn(self.feature_count, n_unsafe)
            # vectors[to_drop[:,unsafe_ind],unsafe_ind] = np.random.randn(n_drop,unsafe_ind.shape[0])

            norm_mask = np.empty((self.feature_count, safe_ind.shape[0]))
            # features to keep have to be normalized
            norm_mask[:, :] = squared[None, :] ** (1. / 2)
            # features to drop are not normalized
            norm_mask[drop_mask[:, safe_ind]] = 1
            vectors[:, safe_ind] *= norm_mask
            full_drop_mask[:, safe_ind] = drop_mask[:, safe_ind]

            if n_unsafe > 0:
                vectors[:, unsafe_ind], full_drop_mask[:, unsafe_ind] = \
                    self.generate_drop_weights_rec(n_unsafe, vectors[:, unsafe_ind],
                        full_drop_mask[:, unsafe_ind], drop_prob * self.drop_decay)

            return vectors, full_drop_mask

    def get_embedding(self, feat_matrix, output_matrix):
        raise NotImplementedError('No full embedding function implemented.')

    def compute_embedding(self, feature_matrix, index_range=None):
        if index_range is None:
            index_range = (0, feature_matrix.shape[1])
        if not self.includes_linear_model:
            embedding = np.empty((self.n_embedding_features, index_range[1] - index_range[0]))
            raw_features = feature_matrix[:, index_range[0]:index_range[1]]
            self.get_embedding(raw_features, embedding)
        else:
            embedding = np.empty((feature_matrix.shape[0] + self.n_embedding_features,
                                 index_range[1] - index_range[0]))
            embedding[:feature_matrix.shape[0], :] = feature_matrix[:, index_range[0]:index_range[1]]
            embedding[feature_matrix.shape[0]:, :] = self.get_embedding(embedding[:
                    feature_matrix.shape[0], :], embedding[
                    feature_matrix.shape[0]:, :])
        return embedding

    def get_partial_embedding(self, feat_matrix, feat_indices):
        raise NotImplementedError('No partial embedding function implemented.')

    def get_test_rankings(self, test_feature_matrix, test_doc_ranges, inverted=True):

        if self.update_embedding:
            if self.test_embeddings is None:
                self.test_embeddings = self.compute_embedding(test_feature_matrix)
            else:
                update_indices = np.where(self.new_features)[0]
                self.test_embeddings[update_indices, :] = \
                    self.get_partial_embedding(test_feature_matrix, update_indices)
            assert self.test_embeddings.shape == (self.feature_count, test_doc_ranges[-1])
            self.update_embedding = False
            self.new_features[:] = 0

        # non interleaved rankings
        return ProbMeanBandit.get_test_rankings(self, self.test_embeddings, test_doc_ranges,
                                                inverted=inverted)

    def get_train_ranking(self, feature_matrix, doc_ranges, ranking_i):

        if not self.store_train_embeddings:
            embedding = self.compute_embedding(feature_matrix, doc_ranges[ranking_i:ranking_i + 2])
        else:
            if self.train_embeddings is None:
                self.train_embeddings = self.compute_embedding(feature_matrix)
            embedding = self.train_embeddings[:, doc_ranges[ranking_i]:doc_ranges[ranking_i + 1]]

        return ProbMeanBandit.get_train_ranking(self, embedding, [0, embedding.shape[1]], 0)

    def update_to_interaction(self, clicks):
        ProbMeanBandit.update_to_interaction(self, clicks)
        if self.enable_drop:
            self.drop_weights()

    def drop_weights(self):
        dropped = np.abs(self.current_best) < 10 ** -7
        if np.sum(dropped) > 0 and self.model_updates > self.min_updates_for_drop:
            drop_ind = np.where(dropped)[0]
            if self.feature_count - len(drop_ind) < self.min_embedding_features:
                np.random.shuffle(drop_ind)
                drop_ind = sorted(drop_ind[:self.feature_count - self.min_embedding_features])
            self.set_message('N_DROP', len(drop_ind))
            if self.permanent_drop:
                self.permanently_drop_weights(drop_ind)
            else:
                self.resample_weights(drop_ind)

    def resample_weights(self, emb_feature_ind):
        raise NotImplementedError('No resampling method implemented.')

    def permanently_drop_weights(self, emb_feature_ind):
        raise NotImplementedError('No permanent dropping of features implemented.')

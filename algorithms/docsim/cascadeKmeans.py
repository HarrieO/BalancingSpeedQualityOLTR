# -*- coding: utf-8 -*-

import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.nonlinear.embeddingbandit import EmbeddingBandit
import numpy as np
from sklearn.cluster import KMeans


class CascadeKMeans(EmbeddingBandit):

    """
    This class picks a subset of features and creates a linear model only using those features.
    """

    def __init__(self, embedding_arguments, mgd_arguments, num_data_features, k, conv_hist,
                 change_threshold, linear_renorm=False):
        EmbeddingBandit.__init__(self, embedding_arguments, mgd_arguments, num_data_features, k)
        self.store_train_embeddings = False
        self._previous_best_models = []
        self._conv_hist = conv_hist
        self._docsim_ranking = True
        self._change_threshold = change_threshold
        self.add_message('UPDATE_DIST', 1)
        self.add_message('MODEL_NORM', 0)
        self._linear_renorm = linear_renorm

    def setup_on_training_data(self, train_feature_matrix, train_doclist_ranges):
        kmeans = KMeans(n_clusters=self.feature_count, n_jobs=1)
        kmeans.fit(train_feature_matrix.T)
        self.support_vectors = kmeans.cluster_centers_.T

    def clean(self):
        self.support_vectors = None
        del self.support_vectors
        gc.collect()

    def sample_candidates(self):
        generated_weights = self.generate_drop_weights(self.n_generate)
        return generated_weights

    def get_embedding(self, feat_matrix, output_matrix):
        if self._docsim_ranking:
            output_matrix[:, :] = np.dot(self.support_vectors.T, feat_matrix)
        else:
            output_matrix[:, :] = feat_matrix

    def get_partial_embedding(self, feat_matrix, feat_indices):
        if self._docsim_ranking:
            return np.dot(self.support_vectors[feat_indices, :].T, feat_matrix)
        else:
            return feat_matrix[feat_indices, :]

    def update_from_winners(self, winners, prefs):
        self._previous_best_models.insert(0, self.current_best.copy())
        if winners.shape[0] > 0:
            n_winners = winners.shape[0]
            self.current_best[:, 0] += self.alpha * np.sum(self.weights[:, winners]
                    - self.current_best[:, 0][:, None], axis=1) / n_winners
        current_model_norm = np.linalg.norm(self.current_best)
        current_dimensionality = self.feature_count
        self.add_message('MODEL_NORM', current_model_norm)
        norms = np.linalg.norm(self._previous_best_models[-1]) * current_model_norm
        if norms != 0:
            change = 1.0 - np.dot(self._previous_best_models[-1].T, self.current_best).flatten()[0] \
                / norms
            self.add_message('UPDATE_DIST', change)
            if change < self._change_threshold and self._docsim_ranking \
                and len(self._previous_best_models) >= self._conv_hist:
                self._docsim_ranking = False
                self.feature_count = self.n_raw_features
                self.n_embedding_features = self.n_raw_features
                docsim_best = self.current_best
                self.current_best = np.zeros((self.feature_count, 1))
                self.current_best[:, 0] = np.dot(docsim_best[:, 0], self.support_vectors.T)
                if self._linear_renorm:
                    self.current_best = self.current_best / np.linalg.norm(self.current_best) \
                        * current_model_norm * current_dimensionality / self.feature_count
                else:
                    self.current_best = self.current_best / np.linalg.norm(self.current_best) \
                        * current_model_norm * (np.sqrt(current_dimensionality)
                                                / np.sqrt(self.feature_count))
                self._previous_best_models = []
                self.new_features = np.ones(self.feature_count)
                self.update_embedding = True
                self.train_embeddings = None
                self.test_embeddings = None
        else:
            self.add_message('UPDATE_DIST', 1.0)
        self._previous_best_models = self._previous_best_models[:self._conv_hist]

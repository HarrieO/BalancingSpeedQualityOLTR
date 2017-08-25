# -*- coding: utf-8 -*-

import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.nonlinear.embeddingbandit import EmbeddingBandit
import numpy as np


class NormalizedSVMBandit(EmbeddingBandit):

    """
    This class picks a subset of features and creates a linear model only using those features.
    """

    def __init__(self, embedding_arguments, mgd_arguments, num_data_features, k, gradient_weight,
                 kernel, sigma=None, poly_c=None, poly_d=2, conv_hist=10):
        EmbeddingBandit.__init__(self, embedding_arguments, mgd_arguments, num_data_features, k)
        self.gradient_weight = gradient_weight
        self.store_train_embeddings = gradient_weight == 0
        self.sigma = sigma
        self.poly_c = poly_c
        self.poly_d = poly_d
        if kernel == 'RBF':
            assert not self.sigma is None, 'No Sigma given for RBF kernel.'
        elif kernel in ['polynomial', 'poly']:
            kernel = 'poly'
            assert not self.poly_c is None, 'No c given for polynomial kernel.'
            assert not self.poly_d is None, 'No d given for polynomial kernel.'
        else:
            assert kernel in ['linear', 'euclidian'], 'Kernel %s unkown.' % kernel
        self.kernel = kernel
        self._previous_best_models = []
        self._conv_hist = conv_hist
        self.add_message('UPDATE_DIST', 1)

    def setup_on_training_data(self, train_feature_matrix, train_doclist_ranges):
        self.vector_indices = np.arange(train_feature_matrix.shape[1])
        np.random.shuffle(self.vector_indices)
        self.vector_indices = self.vector_indices[:self.feature_count]
        self.support_vectors = train_feature_matrix[:, self.vector_indices]
        self.updates_per_vector = np.zeros(self.feature_count)

    def clean(self):
        self.support_vectors = None
        del self.support_vectors
        gc.collect()


    def sample_candidates(self):
        generated_weights = self.generate_drop_weights(self.n_generate)
        return generated_weights

    def get_embedding(self, feat_matrix, output_matrix):
        n_doc = feat_matrix.shape[1]
        if self.kernel == 'linear':
            output_matrix[:,:] = np.dot(self.support_vectors.T, feat_matrix)
        elif self.kernel == 'poly':
            output_matrix[:,:] = (np.dot(self.support_vectors.T, feat_matrix) + self.poly_c) ** self.poly_d
        elif self.kernel == 'RBF':
            # distances = np.empty((self.n_embedding_features, n_doc))
            temp = np.empty((self.n_raw_features, n_doc))
            for i in xrange(self.n_embedding_features):
                temp[:, :] = self.support_vectors[:, i][:, None]
                temp -= feat_matrix
                norms = np.linalg.norm(temp, axis=0)
                # print 'Mean norms', np.mean(norms), np.std(norms)
                output_matrix[i, :] = np.exp(-norms ** 2 / self.sigma ** 2)
                # print 'Meand dist', np.mean(distances[i,:]), np.std(distances[i,:])
            return output_matrix
        elif self.kernel == 'euclidian':
            temp = np.empty((self.n_raw_features, n_doc))
            for i in xrange(self.n_embedding_features):
                temp[:, :] = self.support_vectors[:, i][:, None]
                temp -= feat_matrix
                norms = np.linalg.norm(temp, axis=0)**2
                output_matrix[i, :] = norms
            return output_matrix
        else:
            assert False, 'Unkown kernel %s.' % self.kernel

        # distances = np.repeat(self.support_vectors, feat_matrix.shape[1], axis=1)
        # distances -= np.tile(feat_matrix, (1,self.feature_count))
        # # distances = np.linalg.norm(self.support_vectors[:,:,None] - feat_matrix[:,None,:], axis=0)
        # distances = np.linalg.norm(distances,axis=0)
        # distances = np.exp(-distances**2 / self.sigma**2)
        # return np.reshape(distances,(self.feature_count,feat_matrix.shape[1]))

    def get_partial_embedding(self, feat_matrix, feat_indices):
        n_doc = feat_matrix.shape[1]
        if self.kernel == 'linear':
            return np.dot(self.support_vectors[feat_indices,:].T, feat_matrix)
        elif self.kernel == 'poly':
            return (np.dot(self.support_vectors[feat_indices,:].T, feat_matrix) + self.poly_c) ** self.poly_d
        elif self.kernel == 'RBF':
            distances = np.empty((len(feat_indices), n_doc))
            temp = np.empty((self.n_raw_features, n_doc))
            for temp_i, i in enumerate(feat_indices):
                temp[:, :] = self.support_vectors[:, i][:, None]
                temp -= feat_matrix
                norms = np.linalg.norm(temp, axis=0)
                # print 'Mean norms', np.mean(norms), np.std(norms)
                distances[temp_i, :] = np.exp(-norms ** 2 / self.sigma ** 2)
                # print 'Meand dist', np.mean(distances[i,:]), np.std(distances[i,:])
            return distances
        else:
            assert False, 'Unkown kernel %s.' % self.kernel



    def update_from_winners(self, winners, prefs):
        self._previous_best_models.insert(0, self.current_best.copy())
        if winners.shape[0] > 0:
            n_winners = winners.shape[0]
            self.current_best[:, 0] += self.alpha * np.sum(self.weights[:, winners]
                    - self.current_best[:, 0][:, None], axis=1) / n_winners
            self.updates_per_vector += 1
        if self.model_updates > self.min_updates_for_drop:
            self.current_best[np.abs(self.current_best) < self.gradient_weight] = 0
            self.current_best[:, 0] -= self.gradient_weight * np.sign(self.current_best[:, 0])
        norms = np.linalg.norm(self._previous_best_models[-1]) * np.linalg.norm(self.current_best)
        if norms != 0:
            self.add_message('UPDATE_DIST', 1.0 - np.dot(self._previous_best_models[-1].T,self.current_best).flatten()[0]/norms)
        else:
            self.add_message('UPDATE_DIST', 1.0)
        self._previous_best_models = self._previous_best_models[:self._conv_hist]

    def permanently_drop_weights(self, emb_feature_ind):
        binary_selection = np.ones(self.feature_count, dtype=bool)
        binary_selection[emb_feature_ind] = False

        self.current_best = self.current_best[binary_selection, :]
        self.vector_indices = self.vector_indices[binary_selection]
        self.updates_per_vector = self.updates_per_vector[binary_selection]
        self.support_vectors = self.support_vectors[:,binary_selection]
        if self.train_embeddings is not None:
            self.train_embeddings = self.train_embeddings[binary_selection, :]
        if self.test_embeddings is not None:
            self.test_embeddings = self.test_embeddings[binary_selection, :]


        self.feature_count = np.sum(binary_selection)
        self.n_embedding_features = self.feature_count

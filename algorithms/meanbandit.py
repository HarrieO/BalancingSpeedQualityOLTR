# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rankings import get_score_rankings, get_candidate_score_rankings
from multileaving.interleaving import teamDraftMultileaving, maxIndices
import multileaving.candidatepreselection as cps
import numpy as np
import random


class MeanBandit(object):

    def __init__(self, mgd_arguments, num_data_features, k):
        self.unit = mgd_arguments.unit
        self.alpha = mgd_arguments.alpha
        self.n_cand = mgd_arguments.n_cand
        self.feature_count = num_data_features
        self.current_best = np.zeros((self.feature_count, 1))
        self.last_interleaving = None
        self.n_interactions = 0
        self.model_updates = 0
        self.messages = {}
        self.default_messages = {}

        if mgd_arguments.n_generate < self.n_cand:
            self.n_generate = self.n_cand
        else:
            self.n_generate = mgd_arguments.n_generate

        assert self.n_cand <= self.n_generate

        self.k = k
        self.teams = None
        self.inverted = None
        self.generating_method = mgd_arguments.generating_method
        if self.generating_method == 'factorized':
            assert self.n_generate <= self.feature_count * 2
        assert self.generating_method == 'random' or self.generating_method == 'factorized'

        self.biased = False
        if mgd_arguments.cand_select_method[-2:] == '-B':
            self.biased = True
            self.cand_select_method = mgd_arguments.cand_select_method[:-2]
        elif mgd_arguments.cand_select_method[-2:] == '-U':
            self.cand_select_method = mgd_arguments.cand_select_method[:-2]
        else:
            self.cand_select_method = mgd_arguments.cand_select_method
        assert self.cand_select_method == 'random' or self.cand_select_method == 'CPS' \
            and (self.biased or not self.biased)
        if self.cand_select_method == 'CPS':
            self.history = True
            self.history_list = []
            self.history_len = mgd_arguments.history_len
        else:
            self.history = False
        self.create_candidates()

    def add_message(self, name, default_value=0):
        self.default_messages[name] = default_value

    def set_message(self, name, value):
        self.messages[name] = value

    def get_messages(self):
        messages = self.default_messages.copy()
        messages.update(self.messages)
        return messages

    def setup_on_training_data(self, train_feature_matrix, train_doclist_ranges):
        pass

    def clean(self):
        pass

    def select_candidates(self, generated_weights):
        if generated_weights.shape[1] == self.n_cand:
            return generated_weights
        elif self.cand_select_method == 'random' or len(self.history_list) == 0:
            indices = np.arange(generated_weights.shape[1])
            np.random.shuffle(indices)
            return generated_weights[:, indices[:self.n_cand]]
        elif self.cand_select_method == 'CPS':
            return cps.preselection(self.n_cand, self.biased, generated_weights, self.history_list)

    def create_candidates(self):
        self.weights = np.zeros((self.feature_count, self.n_cand + 1))
        generated_weights = self.generate_weight_matrix()
        self.weights[:, :-1] = self.select_candidates(generated_weights)
        self.weights[:, -1:] = self.current_best

    def generate_weight_matrix(self):
        if self.generating_method == 'random':
            vectors = np.random.randn(self.feature_count, self.n_generate)
            weights = self.current_best[:, 0][:, None] + vectors / (np.sum(np.abs(vectors) ** 2,
                    axis=0) ** (1. / 2))[None, :] * self.unit
            return weights
        elif self.generating_method == 'factorized':
            vectors = np.zeros((self.feature_count, self.n_cand))
            features = np.repeat(np.arange(self.feature_count), 2)
            diffs = np.ones(self.feature_count * 2)
            diffs[self.feature_count:] = -1
            ind = np.arange(self.feature_count * 2)
            np.random.shuffle(ind)
            ind = ind[:self.n_cand]
            vectors[features[ind], np.arange(self.n_cand)] = diffs[ind]
            weights = self.current_best[:, 0][:, None] + vectors
            return weights

    def get_test_rankings(self, test_feature_matrix, test_doc_ranges, inverted=True):
        # non interleaved rankings
        return get_score_rankings(self.current_best, test_feature_matrix, test_doc_ranges,
                                  inverted=inverted)

    def get_train_ranking(self, feature_matrix, doc_ranges, ranking_i):
        # non-interleaved rankings
        self.cand_rankings = get_candidate_score_rankings(self.weights, feature_matrix, doc_ranges,
                ranking_i)
        # let CPS know that we are not using PM
        self.inverted = None
        # candidate rankings are multileaved
        self.ranking, self.teams = self.multileave(self.cand_rankings)
        return self.ranking

    def multileave(self, rankings):
        return teamDraftMultileaving(rankings, self.k)

    def process_clicks(self, clicks, history_event):
        if self.teams is None:
            print 'Clicks were on non-multileaved ranking!'

        if self.history and np.any(clicks):
            if not self.biased:
                qid, clicks, result_list, doclist = history_event[:4]
                if self.inverted is None:
                    inverted = np.zeros(rankings.shape)
                    inverted[np.arange(self.n_cand + 1)[:, None], rankings] = \
                        np.arange(rankings.shape[1])[None, :]
                else:
                    inverted = self.inverted
                sample_prob = cps.probability_of_result_list(result_list, self.inverted)
                history_event = history_event + (sample_prob, )
            self.history_list.append(history_event)
            if len(self.history_list) > self.history_len:
                self.history_list = self.history_list[:-self.history_len]

        self.update_to_interaction(clicks)

        self.n_interactions += 1

        self.create_candidates()

    def update_to_interaction(self, clicks):
        # no update without clicks
        if np.any(clicks):

            total_clicks = np.sum(np.logical_and(np.arange(self.n_cand + 1, dtype=np.int32)[:,
                                  None] == self.teams[None, :], (clicks > 0)[None, :]), axis=1)

            winners = np.argwhere(total_clicks == np.amax(total_clicks)).flatten()

            self.last_interleaving = 0
            if not np.any(self.n_cand == winners):
                n_winners = winners.shape[0]
                self.last_interleaving = n_winners
                self.current_best += (self.alpha * np.sum(self.weights[:, winners]
                                      - self.current_best[:, 0][:, None], axis=1) / n_winners)[:,
                        None]
                self.model_updates += 1

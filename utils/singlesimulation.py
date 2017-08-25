# -*- coding: utf-8 -*-

import random
import time
import numpy as np
from folddata import get_fold_data
from evaluate import get_idcg_list, evaluate, evaluate_ranking
from clicks import *


class SingleSimulation(object):

    def __init__(self, sim_args, output_queue, click_model, datafold):
        self.train_only = sim_args.train_only
        self.n_impressions = sim_args.n_impressions

        self.k = sim_args.k
        self.click_model = click_model
        self.datafold = datafold
        if not self.train_only:
            self.test_idcg_vector = get_idcg_list(self.datafold.test_label_vector,
                                                  self.datafold.test_doclist_ranges, self.k)
        self.train_idcg_vector = get_idcg_list(self.datafold.train_label_vector,
                                               self.datafold.train_doclist_ranges, self.k)

        self.feature_count = self.datafold.num_features
        self.output_list = ['DATA FOLDER ' + str(self.datafold.data_path)]
        self.output_list += ['HELDOUT DATA ' + str(self.datafold.heldout_tag)]
        self.output_list += ['CLICK MODEL ' + self.click_model.get_name()]
        self.output_queue = output_queue

        self.start_lines = sim_args.print_start
        self.print_frequency = sim_args.print_freq
        self.print_all_train = sim_args.all_train
        self.print_feature_count = sim_args.print_feature_count

    def run(self, ranker, direct_print=False, output_key=None):
        starttime = time.time()
        if direct_print:
            for line in self.output_list:
                print line

        starting_prints = self.start_lines
        print_counter = 1

        ranker.setup_on_training_data(self.datafold.train_feature_matrix,
                                      self.datafold.train_doclist_ranges)

        impressions = 0
        for step_i in range(self.n_impressions):
            r_i = random.choice(range(len(self.datafold.train_doclist_ranges) - 1))

            train_ranking = ranker.get_train_ranking(self.datafold.train_feature_matrix,
                    self.datafold.train_doclist_ranges, r_i)[:self.k]
            ranking_labels = \
                self.datafold.train_label_vector[self.datafold.train_doclist_ranges[r_i]:self.datafold.train_doclist_ranges[r_i
                    + 1]]
            impressions += 1

            clicks = self.click_model.generate_clicks(np.array(train_ranking), ranking_labels)

            if not self.train_only and (step_i < starting_prints or print_counter
                                        >= self.print_frequency):

                test_rankings = ranker.get_test_rankings(self.datafold.test_feature_matrix,
                        self.datafold.test_doclist_ranges, inverted=True)

                cur_string = ''
                cur_string += str(impressions)
                cur_string += ' %s:' % self.datafold.heldout_tag
                cur_string += ' %.6f' % np.mean(evaluate(test_rankings,
                                                self.datafold.test_label_vector,
                                                self.test_idcg_vector,
                                                self.datafold.test_doclist_ranges.shape[0] - 1,
                                                self.k))
                cur_string += ' TRAIN: %.6f ' % np.mean(evaluate_ranking(train_ranking,
                        ranking_labels,
                        self.train_idcg_vector[self.datafold.train_doclist_ranges[r_i]], self.k))
                if self.print_feature_count:
                    cur_string += ' N_FEAT: %d ' % ranker.feature_count
                for name, value in ranker.get_messages().items():
                    cur_string += ' %s: %s ' % (name, value)
                cur_string += str(ranker.last_interleaving)

                self.output_list.append(cur_string)
                if direct_print:
                    print cur_string
            elif self.print_all_train:

                cur_string = ''
                cur_string += str(impressions)
                cur_string += ' TRAIN: %.6f ' % np.mean(evaluate_ranking(train_ranking,
                        ranking_labels,
                        self.train_idcg_vector[self.datafold.train_doclist_ranges[r_i]], self.k))
                if self.print_feature_count:
                    cur_string += 'N_FEAT: %d ' % ranker.feature_count
                for name, value in ranker.get_messages().items():
                    cur_string += ' %s: %s ' % (name, value)
                cur_string += str(ranker.last_interleaving)

                self.output_list.append(cur_string)
                if direct_print:
                    print cur_string

            history_event = r_i, clicks, train_ranking, self.datafold.train_feature_matrix[:,
                    self.datafold.train_doclist_ranges[r_i]:self.datafold.train_doclist_ranges[r_i
                    + 1]]
            ranker.process_clicks(clicks, history_event)

            if print_counter >= self.print_frequency:
                print_counter = 0
            print_counter += 1

        r_i = random.choice(range(len(self.datafold.train_doclist_ranges) - 1))
        train_ranking = ranker.get_train_ranking(self.datafold.train_feature_matrix,
                                                 self.datafold.train_doclist_ranges, r_i)[:self.k]
        ranking_labels = \
            self.datafold.train_label_vector[self.datafold.train_doclist_ranges[r_i]:self.datafold.train_doclist_ranges[r_i
                + 1]]
        impressions += 1

        cur_string = ''
        cur_string += str(impressions)
        if not self.train_only:
            test_rankings = ranker.get_test_rankings(self.datafold.test_feature_matrix,
                    self.datafold.test_doclist_ranges, inverted=True)
            cur_string += ' %s:' % self.datafold.heldout_tag
            cur_string += ' %.6f' % np.mean(evaluate(test_rankings,
                                            self.datafold.test_label_vector, self.test_idcg_vector,
                                            self.datafold.test_doclist_ranges.shape[0] - 1, self.k))
        cur_string += ' TRAIN: %.6f ' % np.mean(evaluate_ranking(train_ranking, ranking_labels,
                                                self.train_idcg_vector[self.datafold.train_doclist_ranges[r_i]],
                                                self.k))
        if self.print_feature_count:
            cur_string += ' N_FEAT: %d ' % ranker.feature_count
        for name, value in ranker.get_messages().items():
                    cur_string += ' %s: %s ' % (name, value)
        cur_string += str(ranker.last_interleaving)

        self.output_list.append(cur_string)
        if direct_print:
            print cur_string

        ranker.clean()

        total_time = time.time() - starttime
        seconds = total_time % 60
        minutes = total_time / 60 % 60
        hours = total_time / 3600
        end_string = 'END TIME %d SECONDS %02d:%02d:%02d' % (total_time, hours, minutes, seconds)
        self.output_list.append(end_string)
        if direct_print:
            print end_string

        if output_key is None:
            self.output_queue.put(self.output_list)
        else:
            self.output_queue.put((output_key, self.output_list))

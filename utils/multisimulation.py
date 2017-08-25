# -*- coding: utf-8 -*-

import random
import os
import time
import sharedmem
import numpy as np
from folddata import get_fold_data
from singlesimulation import SingleSimulation
from multiprocessing import Process, Queue
from Queue import Empty
from utils.clicks import get_click_models


class MultiSimulation(object):

    """
    Class designed to manage the multiprocessing of simulations.
    """

    def __init__(self, simulation_arguments):
        self.sim_args = simulation_arguments
        self.num_proc = simulation_arguments.n_processing
        self.click_models = get_click_models(simulation_arguments.click_models)
        self.data_folders = simulation_arguments.data_folders
        self.validation = simulation_arguments.validation
        self.train_only = simulation_arguments.train_only
        if simulation_arguments.max_folds is None:
            self.max_folds = len(self.data_folders)
        else:
            self.max_folds = simulation_arguments.max_folds
        self.read_bin_data = simulation_arguments.read_binarized_data
        self.store_bin_data = simulation_arguments.store_binarized_data_after_read

        self.n_runs = simulation_arguments.n_runs

        self.output_queue = Queue()
        self.single_sims = []
        self.processes = []

        # load first batch to determine feature count
        self.first_data = get_fold_data(self.data_folders[0], self.validation, train_only=self.train_only,
                                        read_from_pickle=self.read_bin_data,
                                        store_pickle_after_read=self.store_bin_data)
        self.feature_count = self.first_data[0].shape[0]
        self.folds_in_mem = 1

    def make_shared(self, numpy_matrix):
        """
        Avoids the copying of Read-Only shared memory.
        """
        if numpy_matrix is None:
            return None
        shared = sharedmem.empty(numpy_matrix.shape, dtype=numpy_matrix.dtype)
        shared[:] = numpy_matrix[:]
        return shared

    def load_simulations(self):
        """
        Generator for different single simulations to run.
        """
        first = True
        for c_i, cur_folder in enumerate(self.data_folders):
            while self.folds_in_mem >= self.max_folds:
                self.update_active()
                self.check_for_output()
            data_tuple = None
            if first:
                data_tuple = self.first_data
                self.first_data = None
                first = False
            else:
                data_tuple = get_fold_data(cur_folder, self.validation, train_only=self.train_only,
                                           read_from_pickle=self.read_bin_data,
                                           store_pickle_after_read=self.store_bin_data)
            if self.num_proc > 1:
                data_tuple = tuple(self.make_shared(matrix) for matrix in data_tuple)

            for c_m in self.click_models:
                yield (c_i, SingleSimulation(self.sim_args, self.output_queue, c_m, cur_folder, data_tuple))

    def load_processes(self, n_runs, rankerclass, ranker_args, ranker_kargs):
        """
        Generator for processes to be started.
        """
        ranker_setup = rankerclass, ranker_args, ranker_kargs
        for fold_i, sim in self.load_simulations():
            for run_i in xrange(n_runs):
                new_proc = Process(target=self.start_run, args=(sim, ranker_setup, run_i))
                self.processes.append((new_proc, fold_i))
                yield new_proc

    def start_run(self, simulation, ranker_setup, seed=0):
        """
        Performs a single run.
        Random functions get different seeds for each process.
        """
        random.seed(os.urandom((time.time(), seed)))
        np.random.seed(os.urandom(int(time.time() + seed * 100 + seed)))
        rankerclass, ranker_args, ranker_kargs = ranker_setup
        ranker = rankerclass(*ranker_args, **ranker_kargs)
        simulation.run(ranker, self.num_proc == 1)

    def update_active(self):
        """
        Checks how many child processes are still active.
        """
        self.processes = [p for p in self.processes if p[0].is_alive()]
        self.folds_in_mem = len(np.unique([p[1] for p in self.processes]))
        self.active = len(self.processes)
        return self.active

    def check_for_output(self, timeout=1):  # 0):
        """
        Prints output for all finished threads
        """
        try:
            while True:
                output_list = self.output_queue.get(timeout=timeout)
                if self.num_proc > 1:
                    print 'RUN', self.display_i
                    self.display_i += 1
                    for line in output_list:
                        print line
                if self.output_queue.empty():
                    break
        except Empty:
            time.sleep(timeout)

    def run(self, rankerclass, *ranker_args, **ranker_kargs):
        starttime = time.time()
        print '--------START--------'
        self.active = 0
        self.display_i = 0
        for process in self.load_processes(self.n_runs, rankerclass, ranker_args, ranker_kargs):
            process.start()
            if self.num_proc == 1:
                print 'RUN', self.display_i
                self.display_i += 1
            while self.update_active() >= self.num_proc:
                self.check_for_output()

        while self.update_active() > 0:
            self.check_for_output()

        self.check_for_output()
        print '--------END--------'
        total_time = time.time() - starttime
        seconds = total_time % 60
        minutes = total_time / 60 % 60
        hours = total_time / 3600
        print 'Total time taken %02d:%02d:%02d' % (hours, minutes, seconds)

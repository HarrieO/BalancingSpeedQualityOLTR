# -*- coding: utf-8 -*-

import argparse
import time


class SimulationArgumentParser(argparse.ArgumentParser):

    def __init__(self, description=None, set_arguments={}):
        self.description = description
        self._initial_set_arguments = dict(set_arguments)
        self.set_arguments = set_arguments
        self.argument_namespaces = {}
        self._namespace_order = []
        self.set_argument_namespace('argparse.ArgumentParser')
        super(SimulationArgumentParser, self).__init__(description=description)
        self._namespace_order = []
        self.set_argument_namespace('Simulation')

        self.add_argument('--n_runs', dest='n_runs', default=125, type=int,
                          help='Number of runs to be simulated over a Dataset.')

        self.add_argument('--n_impr', dest='n_impressions', default=1000, type=int,
                          help='Number of impressions per simulated run.')

        self.add_argument('--vali', dest='validation', action='store_true',
                          help='Use of validation set instead of testset.')

        self.add_argument('--data_folders', dest='data_folders', type=str, required=True,
                          help='Paths to folders where the data-folds are stored.', nargs='+')

        self.add_argument('--output_folder', dest='output_folder', type=str, required=False,
                          help='Path to folders where outputs should be stored, if not given output will be printed.'
                          , default='/zfs/ilps-plex1/slurm/datastore/hooster2/output/fullruns/')

        self.add_argument('--log_folder', dest='log_folder', type=str, required=False,
                          help='Path to folders where run log and errors will be stored.',
                          default='/zfs/ilps-plex1/slurm/datastore/hooster2/logs/')

        self.add_argument('--average_folder', dest='average_folder', type=str, required=False,
                          help='Path to folders where averaged output of runs will be stored.',
                          default='//zfs/ilps-plex1/slurm/datastore/hooster2/output/averaged/')

        self.add_argument('--small_dataset', dest='small_dataset', action='store_false',
                          help='Set true if dataset is small and memory is never a concern.')

        self.add_argument('--click_models', dest='click_models', type=str, required=True,
                          help='Click models to be used.', nargs='+')

        self.add_argument('--print_start', dest='print_start', type=int, required=False,
                          help='The first n steps that will all be printed.', default=1)

        self.add_argument('--print_freq', dest='print_freq', type=int, required=False,
                          help='The number of steps taken before another one is printed after the first batch.'
                          , default=10)

        self.add_argument('--print_output', dest='print_output', action='store_true',
                          help='Set true if outputs should be printed and not stored.')

        self.add_argument('--max_folds', dest='max_folds', type=int, required=False,
                          help='The maximum number of folds that may be loaded at any time, default is unlimited.'
                          , default=None)

        self.add_argument('--n_proc', dest='n_processing', default=1, type=int,
                          help='Max number of work-processes to run in parallel.')

        self.add_argument('--no_run_details', dest='no_run_details', action='store_true',
                          help='Print all run arguments at start of simulation.')

        self.add_argument('--k --n_results', dest='k', default=10, type=int,
                          help='Number of results shown after each query.')

        self.add_argument('--skip_read_bin_data', dest='read_binarized_data', action='store_false')
        self.add_argument('--skip_store_bin_data', dest='store_binarized_data_after_read',
                          action='store_false')

        self.add_argument('--train_only', dest='train_only', action='store_true',
                          help='Only calculate train NDCG.')

        self.add_argument('--all_train', dest='all_train', action='store_false',
                          help='Stop simulation from printing train NDCG at every step.')

        self.add_argument('--print_feat', dest='print_feature_count', action='store_false',
                          help='Makes the simulation print the number of features the model is using.'
                          )

    def reset_arguments(self):
        self.set_arguments = self._initial_set_arguments.copy()

    def set_argument(self, name, value):
        self.set_arguments[name] = value

    def remove_argument(self, name):
        del self.set_arguments[name]

    def set_argument_namespace(self, namespace):
        if namespace not in self.argument_namespaces:
            self._namespace_order.append(namespace)
            self.argument_namespaces[namespace] = []
        self.argument_namespace = namespace

    def add_argument(self, *args, **kargs):
        if (not 'dest' in kargs or kargs['dest'] not in self.set_arguments) and not any(name
                in args[0].replace('--', '').split() for name in self.set_arguments):
            super(SimulationArgumentParser, self).add_argument(*args, **kargs)
        if 'dest' in kargs:
            self.argument_namespaces[self.argument_namespace].append(kargs['dest'])
        else:
            for prefix in ['--', '-']:
                if (args[0])[:len(prefix)] == prefix:
                    self.argument_namespaces[self.argument_namespace].append((args[0])[len(prefix):])
                    break

    def get_simulation_args(self, args):
        return self.get_args(args, 'Simulation')

    def get_args(self, args, argument_namespace):
        result = {}
        dict_args = vars(args)
        for name in self.argument_namespaces[argument_namespace]:
            if name not in self.set_arguments:
                result[name] = dict_args[name]
            elif name in self.set_arguments:
                result[name] = self.set_arguments[name]
        return argparse.Namespace(**result)

    def parse_args_rec(self):
        args = self.parse_args()
        sim_args = self.get_simulation_args(args)
        assert not sim_args.train_only or sim_args.all_train
        output_str = ''
        if not sim_args.no_run_details:
            output_str += 'SIMULATION %s' % time.strftime('%c')
            output_str += '\n%s' % self.description
            output_str += '\n---------------------'
            output_str += '\nSimulation Arguments'
            output_str += '\n---------------------'
            for name, value in vars(sim_args).items():
                output_str += '\n%s %s' % (name, value)
            output_str += '\n---------------------'
        return output_str, args, sim_args

    def parse_all_args(self, ranker_params={}):
        arg_tuple = self.parse_args_rec()
        output_str = arg_tuple[0]
        output_str += '\nOther Arguments'
        output_str += '\n---------------------'
        args = arg_tuple[1]
        for name, value in ranker_params.items():
            if not any(name in vars(p_args) for p_args in arg_tuple[2:]):
                output_str += '\n%s %s' % (name, value)
        for name, value in vars(args).items():
            if not any(name in vars(p_args) for p_args in arg_tuple[2:]):
                assert name not in ranker_params, 'Argument %s in conflict with given parameter.' \
                    % name
                output_str += '\n%s %s' % (name, value)
        output_str += '\n---------------------'
        return (output_str, ) + arg_tuple[1:]

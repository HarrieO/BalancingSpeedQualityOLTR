# -*- coding: utf-8 -*-

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.argparsers.simulationargparser import SimulationArgumentParser


class MGDArgumentParser(SimulationArgumentParser):

    def __init__(self, description=None, set_arguments={}):
        super(MGDArgumentParser, self).__init__(description=description, set_arguments=set_arguments)
        self.set_argument_namespace('MGDArgumentParser')

        self.add_argument('--n_cand', dest='n_cand', required=True, type=int,
                          help='Number of candidates for combating bandits.')

        self.add_argument('--alpha', dest='alpha', type=float, required=False,
                          help='Learning rate for algorithm.', default=0.01)

        self.add_argument('--unit', dest='unit', type=float, required=False,
                          help='Unit distance for candidates.', default=1.0)

        self.add_argument('--factorized', dest='factorized', action='store_true', default=False,
                          help='Turn on factorized candidates generation.')

        self.add_argument('--n_generate', dest='n_generate', default=0, type=int,
                          help='Number of candidates to generate, before selection.')

        self.add_argument('--cand_select', dest='cand_select_method', default='random', type=str,
                          help='Method of candidate selection for multileaving (random/cps).')

        self.add_argument('--history_length', dest='history_len', default=10, type=int,
                          help='Number of past interactions to base Candidate Pre-Selection on.')

    def get_mgd_args(self, args):
        result = self.get_args(args, 'MGDArgumentParser')
        if self.set_arguments.get('factorized', True) and args.factorized:
            result.generating_method = 'factorized'
        else:
            result.generating_method = 'random'
        return result

    def parse_args_rec(self):
        output_str, args, sim_args = super(MGDArgumentParser, self).parse_args_rec()
        mgd_args = self.get_mgd_args(args)
        if not sim_args.no_run_details:
            output_str += '\nMGD Arguments'
            output_str += '\n---------------------'
            for name, value in vars(mgd_args).items():
                output_str += '\n%s %s' % (name, value)
            output_str += '\n---------------------'
        return output_str, args, sim_args, mgd_args

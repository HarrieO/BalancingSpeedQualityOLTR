# -*- coding: utf-8 -*-

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.argparsers.mgdargparser import MGDArgumentParser


class EmbeddingArgumentParser(MGDArgumentParser):

    def __init__(self, description=None, set_arguments={}):
        super(EmbeddingArgumentParser, self).__init__(description=description, set_arguments=set_arguments)
        self.set_argument_namespace('EmbeddingArgumentParser')

        self.add_argument('--permanent_drop', action='store_true', default=False,
                          help='Features are not replaced after being dropped.')

        self.add_argument('--enable_drop', action='store_true',
                          help='Features are dropped if their value is too low.')

        self.add_argument('--n_before_drop', dest='min_updates_for_drop', required=False, type=int,
                          default=100, help='Number of model updates before weights will be dropped.')

        self.add_argument('--min_emb_feat', dest='min_embedding_features', required=False, type=int,
                          default=1, help='Number of minimal embedding features.')

        self.add_argument('--n_emb_feat', dest='n_embedding_features', required=True, type=int,
                          help='Number of embedding features model should use.')

        self.add_argument('--drop_prob', dest='drop_probability', required=False, type=float,
                          help='Probability of an embedding feature being dropped in a candidate.',
                          default=0.0)

        self.add_argument('--drop_decay', dest='drop_decay', required=False, type=float,
                          help='Decay in drop probability after sampling of canidate failed.', default=0.5)

        self.add_argument('--hybrid', dest='add_linear_model', action='store_true', default=False,
                          help='Set true to add the linear features to the embedding features.')

    def get_emb_args(self, args):
        result = self.get_args(args, 'EmbeddingArgumentParser')
        return result

    def parse_args_rec(self):
        output_str, args, sim_args, mgd_args = super(EmbeddingArgumentParser, self).parse_args_rec()
        emb_args = self.get_emb_args(args)
        if not sim_args.no_run_details:
            output_str += '\nEmbedding Arguments'
            output_str += '\n---------------------'
            for name, value in vars(emb_args).items():
                output_str += '\n%s %s' % (name, value)
            output_str += '\n---------------------'
        return output_str, args, sim_args, mgd_args, emb_args

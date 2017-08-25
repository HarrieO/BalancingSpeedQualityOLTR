# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.datasimulation import DataSimulation
from utils.argparsers.embeddingargparser import EmbeddingArgumentParser
from algorithms.docsim.cascadeKmeans import CascadeKMeans
from algorithms.docsim.staticKmeans import StaticKMeans
from algorithms.nonlinear.docsim import DocSim
from algorithms.nonlinear.normSVMbandit import NormalizedSVMBandit
from algorithms.probmeanbandit import ProbMeanBandit

description = 'Run script for Linear Hybrid Probabilistic Multileave Gradient Descent'
parser = EmbeddingArgumentParser(description=description, set_arguments={
    'cand_select_method': 'random',
    'factorized': False,
    'n_generate': None,
    'history_len': None,
    'n_cand': 19,
    'n_embedding_features': 64,
    'permanent_drop': True,
    'drop_probability': 0,
    'drop_decay': 0,
    'enable_drop': False,
    'min_embedding_features': None,
    })

rankers = []
for vec in [50]:
    parser.set_argument('n_embedding_features', vec)
    ranker_params = {'conv_hist': 10}
    arg_str, args, sim_args, mgd_args, emb_args = parser.parse_all_args(ranker_params)

    run_name = 'PMGD19cand'
    rankers.append((arg_str, run_name, ProbMeanBandit, [mgd_args], {}))

    run_name = 'DocSim_StaticKMeans_%dvectors' % vec
    rankers.append((arg_str, run_name, StaticKMeans, [emb_args, mgd_args], ranker_params))

    ranker_params = {'gradient_weight': float(0), 'kernel': 'linear'}
    arg_str, args, sim_args, mgd_args, emb_args = parser.parse_all_args(ranker_params)
    run_name = 'DocSim_linear_%dvectors' % vec
    rankers.append((arg_str, run_name, NormalizedSVMBandit, [emb_args, mgd_args], ranker_params))

    ranker_params = {'conv_hist': 10, 'change_threshold': 0.01, 'linear_renorm': False}
    arg_str, args, sim_args, mgd_args, emb_args = parser.parse_all_args(ranker_params)
    run_name = 'DocSim_cascade%dhist%sthres_linear_%dvectors' % (ranker_params['conv_hist'], ranker_params['change_threshold'], vec)
    rankers.append((arg_str, run_name, DocSim, [emb_args, mgd_args], ranker_params))

    ranker_params = {'conv_hist': 10, 'change_threshold': 0.01, 'linear_renorm': False}
    arg_str, args, sim_args, mgd_args, emb_args = parser.parse_all_args(ranker_params)
    run_name = 'DocSim_cascadeKMeans_%dvectors' % (vec)
    rankers.append((arg_str, run_name, CascadeKMeans, [emb_args, mgd_args], ranker_params))


sim = DataSimulation(sim_args)
sim.run(rankers)

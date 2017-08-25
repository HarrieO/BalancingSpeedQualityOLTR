import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rankings import get_candidate_score_rankings, invert_rankings
import SampleBasedProbabilisticMultileave as pm
import numpy as np


# history should consist of a list of tuples: (qid, clicks, ranking, doclist)
def preselection(n_select, bias, weights, history, rounds=10,
                 cand_per_round=2, tau=None, cps_method="tournament"):
    history_len = len(history)
    rounds = min(rounds, history_len)
    # his_prob = np.zeros(history_len)
    prob_list = []
    for history_event in history:
        _, clicks, result_list, doclist = history_event[:4]
        assert np.any(clicks)
        if not bias:
            # importance sampling
            sample_prob = history_event[4]
        else:
            # log probabilities
            sample_prob = 1.0

        inverted_rankings = get_candidate_score_rankings(weights,
                                                doclist,
                                                np.array([0,
                                                          doclist.shape[1]]),
                                                0,
                                                inverted = True)
        
        clicked_docs = np.where(clicks)[0]

        cur_prob = probability_of_result_list(result_list,
                                              inverted_rankings,
                                              tau)

        prob_list.append((pm.probability_of_list(result_list,
                                                 inverted_rankings,
                                                 clicked_docs),
                          clicked_docs, sample_prob, cur_prob))

    if cps_method == "tournament":
        cand_ind = np.arange(weights.shape[1])
        n_candidates = weights.shape[1]
        while n_candidates > n_select:
            pref = np.zeros((cand_per_round, cand_per_round))
            np.random.shuffle(cand_ind)
            round_cand = cand_ind[:cand_per_round]
            for _ in xrange(rounds):
                h_i = np.random.randint(history_len)
                probs, clicked_docs, sample_prob, cur_prob = prob_list[h_i]
                probs = probs[:, round_cand]
                probs /= np.sum(probs, axis=1)[:, None]
                pref += pm.preferences_of_list(probs, 1000)*cur_prob/sample_prob
            n_fighters = round_cand.shape[0]
            while n_fighters > 1:
                fighter = np.random.randint(round_cand.shape[0])
                loser = np.argmin(pref[:, fighter])
                cand_ind = cand_ind[cand_ind != round_cand[loser]]
                round_cand = round_cand[round_cand != round_cand[loser]]
                n_fighters -= 1
                n_candidates -= 1
        return weights[:, cand_ind]
    elif cps_method == "direct":
        n_candidates = weights.shape[1]
        pref = np.zeros((n_candidates, n_candidates))
        for probs, _, sample_prob, cur_prob in prob_list[-rounds:]:
            probs /= np.sum(probs, axis=1)[:, None]
            pref += pm.preferences_of_list(probs, 1000) * cur_prob / sample_prob
        return weights[:, np.argsort(np.sum(pref, axis=1))[:n_select]]


def probability_of_result_list(result_list, inverted_rankings, tau=None):
    if tau is None:
        return pm.probability_of_result_list(result_list, inverted_rankings)
    else:
        return pm.probability_of_result_list(result_list, inverted_rankings, tau)
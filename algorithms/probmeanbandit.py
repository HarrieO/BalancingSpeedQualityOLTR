import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import multileaving.SampleBasedProbabilisticMultileave as pm
from utils.rankings import invert_rankings
import numpy as np
from meanbandit import MeanBandit

class ProbMeanBandit(MeanBandit):

    def __init__(self, mgd_arguments, num_data_features, k, exploration=True):
        self.exploration = exploration
        MeanBandit.__init__(self, mgd_arguments, num_data_features,k)

    def multileave(self,rankings,already_inverted=False):
        if not already_inverted:
            self.inverted = invert_rankings(rankings) 
        if self.exploration:   
            self.ranking = pm.multileave(self.inverted, self.k)
        else:
            self.ranking = rankings[-1,:]
        return self.ranking, True

    def update_to_interaction(self,clicks):
        self.last_interleaving = 0
        # no update without clicks
        if np.any(clicks):
            click_ids = self.ranking[np.nonzero(clicks)[0]]
            probs = pm.probability_of_list(self.ranking, self.inverted, click_ids)

            prefs = pm.preferences_of_list(probs,10000)

            winners = np.arange(self.n_cand+1)[prefs[:,-1] > 0]
            n_winners = winners.shape[0]
            self.last_interleaving = n_winners

            self.update_from_winners(winners,prefs)
            
            if winners.shape[0] > 0:
                self.model_updates += 1
            
    def update_from_winners(self, winners, prefs):    
        if winners.shape[0] > 0:
            n_winners = winners.shape[0]
            self.last_interleaving = n_winners
            self.current_best[:,0] += self.alpha*np.sum((self.weights[:,winners]
                                      - self.current_best[:,0][:,None]),axis=1)/n_winners

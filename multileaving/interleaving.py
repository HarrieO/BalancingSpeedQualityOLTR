import random
import numpy as np

def nextIndexToAdd(interResult, ranking, index ):
    while index < len(ranking) and ranking[index] in interResult:
        index += 1
    return index


def teamDraftInterleaving(ranking0, ranking1, k=10):
    # the interleaved result and team-assignment
    result = []
    teams  = []

    k = min(k,min(len(ranking0),len(ranking1)))

    index = 0
    while index < k and ranking0[index] == ranking1[index]:
        result.append(ranking0[index])
        teams.append(-1)
        index += 1

    index0 = len(result)
    index1 = len(result)
    added1 = 0
    added2 = 0
    while len(result) < k:
        if added1 < added2 or (added1 == added2 and random.random() <= 0.5):
            result.append(ranking0[index0])
            teams.append(0)
            added1 += 1
        else:
            result.append(ranking1[index1])
            teams.append(1)
            added2 += 1
        index0 = nextIndexToAdd(result, ranking0, index0)
        index1 = nextIndexToAdd(result, ranking1, index1)

    return result, teams

def minIndices(values):
    indic = [0]
    for i in range(1,len(values)):
        if values[indic[0]] > values[i]:
            indic = [i]
        elif values[indic[0]] == values[i]:
            indic.append(i)
    return indic

def maxIndices(values):
    indic = [0]
    for i in range(1,len(values)):
        if values[indic[0]] < values[i]:
            indic = [i]
        elif values[indic[0]] == values[i]:
            indic.append(i)
    return indic

def moreToAdd(rankings, indices):
    for i, l in enumerate(indices):
        if len(rankings[i]) <= l:
                return False
    return True


def next_index_to_add(inter_result, inter_n, ranking, index):
    while index < ranking.shape[0] and np.any(ranking[index] == inter_result[:inter_n]):
        index += 1
    return index

def teamDraftMultileaving(rankings, k=10):

    n_rankings = rankings.shape[0]
    k = min(k,rankings.shape[1])
    teams = np.zeros(k,dtype=np.int32)
    multileaved = np.zeros(k,dtype=np.int32)

    multi_i = 0
    while multi_i < k and np.all(rankings[1:,multi_i]==rankings[0,multi_i]):
        multileaved[multi_i] = rankings[0][multi_i]
        teams[multi_i] = -1
        multi_i += 1

    indices  = np.zeros(n_rankings,dtype=np.int32) + multi_i
    assign_i = n_rankings
    while multi_i < k:
        if assign_i == n_rankings:
            assignment = np.arange(n_rankings,dtype=np.int32)
            np.random.shuffle(assignment)
            assign_i = 0

        rank_i = assignment[assign_i]
        indices[rank_i] = next_index_to_add(multileaved, multi_i, rankings[rank_i,:], indices[rank_i])
        multileaved[multi_i] = rankings[rank_i,indices[rank_i]]
        teams[multi_i] = rank_i
        indices[rank_i] += 1
        multi_i += 1
        assign_i += 1

    return multileaved, teams


if __name__ == '__main__':
    ranking0 = range(10)
    random.shuffle(ranking0)
    ranking0 = ranking0[:10]
    ranking1 = range(10)
    random.shuffle(ranking1)
    ranking1 = ranking1[:10]
    ranking2 = range(10)
    random.shuffle(ranking2)
    ranking2 = ranking2[:10]
    print ranking0
    print ranking1
    print ranking2
    res, teams = teamDraftMultileaving([ranking0,ranking1,ranking2])
    print teams
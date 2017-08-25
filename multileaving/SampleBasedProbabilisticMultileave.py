import numpy as np

def probability_of_result_list(result_list, inverted_rankings, tau=3.0):
    '''
    ARGS: (all np.array of docids)
    - result_list: the multileaved list
    - inverted_rankings: dtype:float matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x
    - clicked_docs: the indices of the clicked_docs

    RETURNS
    - probability of the result_list being created by the rankers
    '''
    n = inverted_rankings.shape[1]
    # normalization denominator for the complete ranking
    sigmoid_total = np.sum(float(1) / (np.arange(n) + 1) ** tau)*inverted_rankings.shape[0]

    # make sure inverted rankings is of dtype float
    # the unnormalized distribution over the probability of each document being added (as first document)
    sigmas = np.sum(1/(inverted_rankings[:,result_list]+1.)**tau,axis=0)

    # cumsum is used to renormalize the probs, it contains the part
    # the denominator that has to be removed (due to previously added docs)
    cumsum = np.zeros(sigmas.shape)
    cumsum[1:] = np.cumsum(sigmas[:-1])

    probs = sigmas / (sigmoid_total - cumsum)
    
    return np.prod(probs)

def multileave(inverted_rankings,k,tau=3.0):
    '''
    ARGS: (all np.array of docids)
    - inverted_rankings: matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x

    RETURNS
    - ranking of indices corresponding to inverted_rankings
    '''

    n = inverted_rankings.shape[1]
    k = min(n,k)

    assignments = np.random.randint(0,inverted_rankings.shape[0],k)

    denominator = np.zeros((k)) + np.sum(float(1) / (np.arange(n) + 1) ** tau) 
    probs = 1./(inverted_rankings[assignments,:]+1)**tau

    ranking = np.zeros(k,dtype=np.int32)

    docids = np.arange(n)

    for i in range(k):
        upper = np.cumsum(probs[i,:])
        lower = np.zeros(upper.shape)
        lower[1:] += upper[:-1]

        coinflip = np.random.rand()

        logic = np.logical_and(lower/denominator[i] < coinflip,
                               upper/denominator[i] >= coinflip)

        raw_i = np.where(logic)[0][0]

        ranking[i] = docids[raw_i]

        docids[raw_i:-1] = docids[raw_i+1:]

        denominator -= probs[:,raw_i]
        if raw_i < n-1:
            probs[:,raw_i:-1] = probs[:,raw_i+1:]

    return ranking

def tied_ranking(scores,n_labels=None):

    if n_labels == None:
        n_labels = scores.shape[1]

    # sort scores per row
    sorted_ind = np.argsort(scores)
    sorted_scores = scores[np.arange(scores.shape[0])[:,None],sorted_ind]
    
    # determine at what places scores change (thus the rank increases)
    perimeters = sorted_scores[:,1:] != sorted_scores[:,:-1]
    ranks = np.zeros(scores.shape,dtype=np.int32)
    # a cumsum can thus determine the rank at each position
    ranks[:,1:] += np.cumsum(perimeters,axis=1)
    
    # NOTE TO SELF: 3d matrices can probably be avoided everywhere by 'clever' bincounting
    counts = np.sum(ranks[:,:,None] == np.arange(n_labels)[None,None,:],axis=1)
    upper = np.cumsum(counts,axis=1)
    lower = np.zeros(counts.shape,dtype=np.int32)
    lower[:,1:] = upper[:,:-1]

    ind = ranks[:,:,None] == np.arange(n_labels)[None,None,:]
    mapping = np.zeros(ranks.shape+(n_labels,),dtype=np.int32)
    mapping[ind] = (np.zeros(ranks.shape+(n_labels,),dtype=np.int32)+ counts[:,None,:])[ind]
    unmapped_counts = np.sum(mapping,axis=2)
    
    mapping[ind] = (np.zeros(ranks.shape+(n_labels,),dtype=np.int32)+ lower[:,None,:])[ind]
    unmapped_lower = np.sum(mapping,axis=2)

    mapping[ind] = (np.zeros(ranks.shape+(n_labels,),dtype=np.int32)+ upper[:,None,:])[ind]
    unmapped_upper = np.sum(mapping,axis=2)

   
    row_ind = np.arange(scores.shape[0])[:,None]
    mapped_ranks = np.zeros(scores.shape,dtype=np.int32)
    mapped_ranks[row_ind,sorted_ind] = ranks+1
    mapped_lower = np.zeros(scores.shape,dtype=np.int32)
    mapped_lower[row_ind,sorted_ind] = unmapped_lower
    mapped_upper = np.zeros(scores.shape,dtype=np.int32)
    mapped_upper[row_ind,sorted_ind] = unmapped_upper
    mapped_counts = np.zeros(scores.shape,dtype=np.int32)
    mapped_counts[row_ind,sorted_ind] = unmapped_counts

    return mapped_ranks, mapped_counts, mapped_lower, mapped_upper
    
def probability_of_list_with_ties(tied_statistics, result_list, clicked_docs, tau=3.0):

    ranks, counts, lower, upper = tied_statistics

    n = counts.shape[1]
    sigmoid = np.zeros(n+1)
    sigmoid[1:] = np.cumsum(float(1) / (np.arange(n) + 1.) ** tau)
    
    sigmas = ((sigmoid[upper]-sigmoid[lower])/counts)

    #cumsum is used to renormalize the probs, it contains the part
    # the denominator that has to be removed (due to previously added docs)
    cumsum = np.zeros(counts.shape)
    cumsum[:,result_list[1:]] = np.cumsum(sigmas[:,result_list[:-1]],axis=1)
    
    clicked_probs = sigmas[:,clicked_docs]
    clicked_probs /= sigmoid[-1] - cumsum[:,clicked_docs]

    return (clicked_probs / np.sum(clicked_probs,axis=0)[None,:]).T

    

def probability_of_list(result_list, inverted_rankings, clicked_docs, tau=3.0):
    '''
    ARGS: (all np.array of docids)
    - result_list: the multileaved list
    - inverted_rankings: matrix (rankers x documents) where [x,y] corresponds to the rank of doc y in ranker x
    - clicked_docs: the indices of the clicked_docs

    RETURNS
    -sigmas: matrix (rankers x clicked_docs) with probabilty ranker added clicked doc
    '''
    n = inverted_rankings.shape[1]
    # normalization denominator for the complete ranking
    sigmoid_total = np.sum(float(1) / (np.arange(n) + 1) ** tau)

    #cumsum is used to renormalize the probs, it contains the part
    # the denominator that has to be removed (due to previously added docs)
    cumsum = np.zeros(inverted_rankings.shape)
    cumsum[:,result_list[1:]] = np.cumsum(
        (float(1)/(inverted_rankings[:,result_list[:-1]]+1.)**tau),
        axis=1)

    # make sure inverted rankings is of dtype float
    sigmas = 1/(inverted_rankings[:,clicked_docs].T+1.)**tau
    sigmas /= sigmoid_total - cumsum[:,clicked_docs].T
    
    return (sigmas / np.sum(sigmas,axis=1)[:,None])

def preferences_of_list(probs,n_samples):
    '''
    ARGS:
    -probs: clicked docs x rankers matrix with probabilities ranker added clicked doc  (use probability_of_list)
    -n_samples: number of samples to base preference matrix on

    RETURNS:
    - preference matrix: matrix (rankers x rankers) in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
      the value is analogous to the (average) degree of preference
    '''
    n_clicks = probs.shape[0]
    n_rankers = probs.shape[1]
    #determine upper bounds for each ranker (to see prob distribution as set of ranges)
    upper = np.cumsum(probs,axis=1)

    #determine lower bounds
    lower = np.zeros(probs.shape)
    # lower[:,0] = 0
    lower[:,1:] += upper[:,:-1]
    
    # flip coins, coins fall between lower and upper
    coinflips = np.random.rand(n_clicks,n_samples)
    # make copies for each sample and each ranker
    comps = coinflips[:,:,None]
    # determine where each coin landed
    log_assign = np.logical_and(comps > lower[:,None,:],comps <= upper[:,None,:])
    # click count per ranker (samples x rankers)
    click_count = np.sum(log_assign,axis=0)
    # the preference matrix for each sample
    prefs = np.sign(click_count[:,:,None]-click_count[:,None,:])

    # the preferences are averaged for each pair
    # in this matrix [x,y] > 0 means x won over y and [x,y] < 0 means x lost from y
    # the value is analogous to the (average) degree of preference
    return np.sum(prefs,axis=0)/float(n_samples)


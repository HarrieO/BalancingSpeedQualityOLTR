import numpy as np
import os.path, gc

FOLDDATA_WRITE_VERSION=1

def read_file(path, all_features={}):
    '''
    Read letor file and returns dict for qid to indices, labels for queries
    and list of doclists of features per doc per query.
    '''
    current_qid = None
    queries  = {}
    queryIndex = 0
    doclists = []
    labels   = []
    for line in open(path, "r"):
        info = line[:line.find("#")].split()

        qid = info[1].split(":")[1]
        label = int(info[0])
        if qid not in queries:
            queryIndex   = len(queries)
            queries[qid] = queryIndex
            doclists.append([])
            labels.append([])
            current_qid = qid
        elif qid != current_qid:
            queryIndex = queries[qid]
            current_qid = qid

        featureDict = {}
        for pair in info[2:]:
            featid, feature = pair.split(":")
            all_features[featid] = True
            featureDict[featid] = float(feature)
        doclists[queryIndex].append(featureDict)
        labels[queryIndex].append(label)

    return queries, doclists, labels, all_features

def create_feature_mapping(feature_dict):
    total_features = 0
    feature_map = {}
    for fid in feature_dict:
        if fid not in feature_map:
            feature_map[fid] = total_features
            total_features  += 1
    return feature_map

def convert_featureDicts(doclists,label_lists,feature_mapping, query_level_norm=True):
    """
    represents doclists/features as matrix and list of ranges
    """
    total_features = len(feature_mapping)
    total_docs     = 0
    ranges         = []
    for doclist in doclists:
        start_range = total_docs
        total_docs += len(doclist)
        ranges.append((start_range,total_docs))

    feature_matrix = np.zeros((total_features, total_docs))
    label_vector = np.zeros((total_docs), dtype=np.int32)

    new_doclists = None
    index = 0
    for doclist, labels in zip(doclists,label_lists):
        start = index
        for featureDict,label in zip(doclist,labels):
            for fid, value in featureDict.items():
                if fid in feature_mapping:
                    feature_matrix[feature_mapping[fid],index] = value
            label_vector[index] = label
            index += 1
        end = index
        if query_level_norm:
            feature_matrix[:,start:end] -= np.amin(feature_matrix[:,start:end],axis=1)[:,None]
            safe_max = np.amax(feature_matrix[:,start:end],axis=1)
            safe_ind = safe_max != 0
            feature_matrix[safe_ind,start:end] /= (safe_max[safe_ind])[:,None]

    qptr = np.zeros(len(ranges)+1, dtype=np.int32)
    for i, ra in enumerate(ranges):
        qptr[i+1] = ra[1]

    return feature_matrix, qptr, label_vector

def get_fold_data(folder, validation_as_test=False, train_only=False, store_pickle_after_read=True,
                  read_from_pickle=True):
    """
    Returns data from a fold folder (letor format)
    """
    # clear any previous datasets
    gc.collect()

    train_read = False
    test_read = False
    if validation_as_test:
        train_pickle_name = "binarized_train_val.npz"
        test_pickle_name = "binarized_val.npz"
    else:
        train_pickle_name = "binarized_train.npz"
        test_pickle_name = "binarized_test.npz"

    train_pickle_path = folder+train_pickle_name
    test_pickle_path = folder+test_pickle_name
    if read_from_pickle:
        if os.path.isfile(train_pickle_path):
            loaded_data = np.load(train_pickle_path)
            if 'train_version' in loaded_data and loaded_data['train_version'] == FOLDDATA_WRITE_VERSION:
                feature_map    = loaded_data['feature_map'][()]
                feature_matrix = loaded_data['feature_matrix']
                doclist_ranges = loaded_data['doclist_ranges']
                label_vector   = loaded_data['label_vector']
                train_read = True
            del loaded_data
            gc.collect()
        if os.path.isfile(test_pickle_path):
            loaded_data = np.load(test_pickle_path)
            if 'test_version' in loaded_data and loaded_data['test_version'] == FOLDDATA_WRITE_VERSION:
                test_feature_matrix = loaded_data['test_feature_matrix']
                test_doclist_ranges = loaded_data['test_doclist_ranges']
                test_label_vector   = loaded_data['test_label_vector']
                test_read = True
            # remove potentially memory intensive variables
            del loaded_data
            gc.collect()


    if not train_read:
        doclists = []
        labels   = []
        _, n_doclists, n_labels, training_features = read_file(folder+"train.txt")
        doclists.extend(n_doclists)
        labels.extend(  n_labels)

        if not validation_as_test:
            _, n_doclists, n_labels, training_features = read_file(folder+"vali.txt",training_features)
            doclists.extend(n_doclists)
            labels.extend(  n_labels)

        feature_map = create_feature_mapping(training_features)

        feature_matrix, doclist_ranges, label_vector = convert_featureDicts(doclists,labels,feature_map)
        del doclists,labels
        # invoking garbage collection, to avoid memory clogging
        gc.collect()
        if store_pickle_after_read:
            np.savez(train_pickle_path, train_version=FOLDDATA_WRITE_VERSION,
                                        feature_map=feature_map,
                                        feature_matrix=feature_matrix,
                                        doclist_ranges=doclist_ranges,
                                        label_vector=label_vector)
    
    if not train_only and not test_read:
        if not validation_as_test:
            _, test_doclists, test_labels, _ = read_file(folder+"test.txt")
        else:
            _, test_doclists, test_labels, _ = read_file(folder+"vali.txt")

        test_feature_matrix, test_doclist_ranges, test_label_vector = convert_featureDicts(test_doclists, test_labels,feature_map)
        del test_doclists,test_labels
        # invoking garbage collection, to avoid memory clogging
        gc.collect()
        if store_pickle_after_read:
            np.savez(test_pickle_path, test_version=FOLDDATA_WRITE_VERSION,
                                       test_feature_matrix=test_feature_matrix,
                                       test_doclist_ranges=test_doclist_ranges,
                                       test_label_vector=test_label_vector)
    elif train_only:
        test_feature_matrix, test_doclist_ranges, test_label_vector = None,None,None
    return feature_matrix, doclist_ranges, label_vector, test_feature_matrix, test_doclist_ranges, test_label_vector

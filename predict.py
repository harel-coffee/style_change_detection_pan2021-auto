import re
import os
import string
import argparse
import sys
import numpy as np
import math
from tqdm.auto import tqdm
import pickle
from utills import get_files_in_dir, get_file_nums
from features import prepare_entry
import json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


MODEL_FILE = 'temp_data/model.p'


def assign_ordered_ids(in_array):
    out_array = np.zeros(len(in_array)).astype(int)
    ordered_ids = set()
    next_id = 1
    for i, c_id in enumerate(in_array):
        if c_id not in ordered_ids:
            out_array[np.argwhere(in_array==c_id)[:, 0]] = next_id
            ordered_ids.add(c_id)
            next_id += 1
    return out_array.tolist()


def make_predictions(paras, transformer, primary_scaler, secondary_scaler, clf, clust_thresh, return_probs=False):
    X = transformer.transform(paras).todense()
    X = primary_scaler.transform(X)
    X[np.isnan(X)] = 0
    s = 0
    e = len(paras)

    X1 = X[s:(e-1)]
    X2 = X[s+1:e]
    X_difference = np.abs(X2 - X1)
    p = clf.predict_proba(secondary_scaler.transform(X_difference))[:,1]

    prob_multi_author = p.mean()
    task1_sol = int(prob_multi_author > 0.5)
    task2_sol = [int(pp > 0.5) for pp in p]

    all_paragraphs_i = len(X)
    X_all = X[0:all_paragraphs_i] 

    idxs = []
    for i in range(0, len(X) - 1):
        for j in range(i + 1, len(X)):
            idxs.append([i, j])

    idxs = np.array(idxs)
    X_diff = secondary_scaler.transform(np.abs(X[idxs[:, 0]] - X[idxs[:, 1]]))
    probs =  clf.predict_proba(X_diff)[:, 1]
    
    '''
    n = len(paras)
    dist_sq = np.zeros((n, n))
    for i, (a, b) in enumerate(idxs):
        dist_sq[a, b] = probs[i]
        dist_sq[b, a] = probs[i]'''
    Z = linkage(probs, 'ward')
    paragraph_author = list(fcluster(Z, t=clust_thresh, criterion='distance')) 
    ctr_enter_if = 0
    if len(np.unique(paragraph_author)) > 4:
        ctr_enter_if += 1 
        paragraph_author = list(fcluster(Z, t=4, criterion='maxclust'))
        
    result = [int(task1_sol), task2_sol, assign_ordered_ids(paragraph_author)]
    
    if return_probs:
        result.extend([p, prob_multi_author, idxs, probs])
        
    result.append(ctr_enter_if)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction Script: PAN 2021')
    parser.add_argument('-i', type=str,
                        help='Input dir')
    parser.add_argument('-o', type=str, 
                        help='Output dir')
    args = parser.parse_args()
    
    # validate:
    if not args.i:
        raise ValueError('Input dir path is required')
    if not args.o:
        raise ValueError('Output dir path is required')
        
    INPUT_DIR = args.i
    OUTPUT_DIR = args.o
    
    with open(MODEL_FILE, 'rb') as f:
        transformer, primary_scaler, secondary_scaler, clf = pickle.load(f)

    input_files = get_files_in_dir(INPUT_DIR, file_base='problem')
    print('Number of input files:', len(input_files), flush=True)
    
    for f in tqdm(input_files):
        with open(f, 'r') as input_file:
            paras = [prepare_entry(line.strip(), mode='accurate', tokenizer='casual') for line in input_file.readlines()]
            #paras = [prepare_entry(line.strip(), mode='accurate', tokenizer='casual') for line in input_file.read().split('\n\n')]
            r = make_predictions(paras, transformer, primary_scaler, secondary_scaler, clf, clust_thresh=0.5)
            solution = {
                "multi-author": r[0],
                "changes": r[1],
                "paragraph-authors": r[2]
            }
            num = re.findall(r'\d+', os.path.basename(f))[0]
            with open(os.path.join(OUTPUT_DIR, 'solution-problem-'+num+'.json'), 'w') as fout:
                json.dump(solution, fout)
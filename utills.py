import itertools
import numpy as np
import glob
import os
import json



def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=int)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def binarize(y, threshold=0.5):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will always be `0` or `1`.
    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)
    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    """

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def get_files_in_dir(file_path, file_base=''):
    """
        Given the file path/directory containing preprocess file path
        Return an array containing the name of all files in the path
    """
    return glob.glob(os.path.join(file_path, file_base + '*'))

def get_file_nums(file_name_arr):
    """
        Given an array of all file names,
        return an array, nums, that list the order of the files.
    """
    prob_num = [f.split('/')[-1] for f in file_name_arr]
    problem_nums = [p.split('.') for p in prob_num]
    nums = [p[0].split('-')[-1] for p in problem_nums]
    
    return nums

def load_preprocessed_paras(file_names_arr):
    """
        Given array of pre-process file names,
        Return a matrix with all the pre-processed paragraphs.
    """
    pre_proc_arr = []

    for name in file_names_arr:
        with open(name, "r") as f:
            for l in f:
                d = json.loads(l)
                pre_proc_arr.append(d)
                
    return pre_proc_arr

def paragraph_indecies(filenames_arr):
    """
        Given array of pre-process file names,
        Return a dictionary with filename: (first precocessed paragraph,last preprocessed paragraph + 1).
    """  
    s = 0
    e = 0
    file_paragraph_indecies = {}

    for name in filenames_arr:
        with open(name, "r") as f:
            for l in f:
                e += 1
            file_paragraph_indecies[name] = (s,e)
            s = e

    return file_paragraph_indecies

def order_files_multi_years(file_name_arr, yr_str):
    """
        Given an array of all file names,
        return an array, nums, that list the order of the files.
    """
    prob_num = [f.split('/')[-1] for f in file_name_arr]
    problem_nums = [p.split('.') for p in prob_num]
    nums = [p[0].split('-')[-1]+"_"+yr_str for p in problem_nums]
    
    return nums

def load_ground_truth_from_nums(base_path, nums):
    result = []
    for n in nums:
        result.append(json.load(open(os.path.join(base_path, 'truth-problem-' + n + '.json'), 'r')))
    return result
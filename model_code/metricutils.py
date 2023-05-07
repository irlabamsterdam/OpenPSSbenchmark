import scipy
import json
import numpy as np
import pandas as pd
from typing import Dict, Union
from collections import defaultdict
from nltk.metrics import windowdiff
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None


def read_json(filename: str) -> dict:
    """
    This function provides a convenience function for reading in a file
    that is in the json standard.
    :param filename: string specifying the path to the json file
    :return: JSON-style dict with the contents from the file
    """
    with open(filename, 'r') as json_file:
        contents = json.load(json_file)
    return contents


def get_ground_truth_from_dataframe(dataframe: pd.DataFrame, col: str) -> Dict[str, list]:
    """
    This function takes as input the test dataframe, and return a dictionary
    with stream names as keys and the gold standard streams in
    binary vector format.

    """
    out = {}
    for doc_id, content in dataframe.groupby('name'):
        out[doc_id] = [int(item) for item in content[col].tolist()]
    return out


def length_list_to_bin(list_of_lengths: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param list_of_lengths:  containing the lengths of the individual documents
    in a stream as integers.
    :return: list representing the stream in binary format.
    """

    if not all([item > 0 for item in list_of_lengths]):
        raise ValueError

    # Set up the output array
    out = np.zeros(shape=(sum(list_of_lengths)))

    # First document is always a boundary
    out[0] = 1

    # if only one document return the current representation
    if len(list_of_lengths) == 1:
        if type(list_of_lengths) == list:
            return out.tolist()
        else:
            return out

    # Boundaries are at the cumulative sums of the number of pages
    # >>> doc_list = [2, 4, 3, 1]
    # >>> np.cumsum(doc_list) -> [2 6 9]

    # [:-1] because last document has boundary at end of array
    out[np.cumsum(list_of_lengths[:-1])] = 1
    if type(list_of_lengths) == list:
        return out.tolist()
    else:
        return out


def bin_to_length_list(binary_vector: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param binary_vector: np array containing the stream of pages
    in the binary format.
    :return: A numpy array representing the stream as a list of
    document lengths.
    """

    # make sure the vector only contains 1s and zeros
    if not all([item in [0, 1] for item in binary_vector]):
        raise ValueError

    return_type = type(binary_vector)

    if type(binary_vector) == list:
        binary_vector = np.array(binary_vector)

    # We retrieve the indices of the ones with np.nonzero
    bounds = binary_vector.nonzero()[0]

    # We add the length of the array so that it works
    # with ediff1d, as this get the differences between
    # consecutive elements, and otherwise we would miss
    # the list document.
    bounds = np.append(bounds, len(binary_vector))

    # get consecutive indices
    out = np.ediff1d(bounds)

    if return_type == list:
        return out.tolist()
    else:
        return out


def window_diff(gold: np.array, prediction: np.array) -> float:

    assert len(gold) == len(prediction)

    gold[0] = 1
    prediction[0] = 1

    k = int(bin_to_length_list(gold).mean()*0.5)
    # small check, in case of a singleton cluster, k will be too large
    if k > len(gold):
        k = len(gold)

    string_gold = ''.join(str(item) for item in gold.astype(int).tolist())
    string_prediction = ''.join(str(item) for item in prediction.astype(int).tolist())

    return windowdiff(string_gold, string_prediction, k=k)


def f1(gold: np.array, prediction: np.array) -> float:
    return f1_score(gold, prediction)


def precision(gold: np.array, prediction: np.array) -> float:
    return precision_score(gold, prediction)


def recall(gold: np.array, prediction: np.array) -> float:
    return recall_score(gold, prediction)


def calculate_metrics_one_stream(gold_vec, prediction_vec):
    out = {}

    gold_vec = np.array(gold_vec)
    prediction_vec = np.array(prediction_vec)

    prediction_vec[0] = 1
    scores = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
              'Boundary': f1(gold_vec, prediction_vec),
              'WindowDiff': window_diff(gold_vec, prediction_vec)}

    scores_precision = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
                        'Boundary': precision(gold_vec, prediction_vec),
                        'WindowDiff': window_diff(gold_vec, prediction_vec)}

    scores_recall = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
                     'Boundary': recall(gold_vec, prediction_vec),
                     'WindowDiff': window_diff(gold_vec, prediction_vec)}

    out['precision'] = scores_precision
    out['recall'] = scores_recall
    out['F1'] = scores

    return out


def calculate_scores_df(gold_standard_dict, prediction_dict):
    all_scores = defaultdict(dict)
    for key in gold_standard_dict.keys():
        metric_scores = calculate_metrics_one_stream(gold_standard_dict[key],
                                                     prediction_dict[key])
        for key_m in metric_scores.keys():
            all_scores[key_m][key] = metric_scores[key_m]
    return {key: pd.DataFrame(val) for key, val in all_scores.items()}


def calculate_mean_scores(gold_standard_dict, prediction_dict,
                          show_confidence_bounds=True):
    scores_df = {key: val.T.mean().round(2) for key, val in
                 calculate_scores_df(gold_standard_dict,
                                     prediction_dict).items()}
    scores_combined = pd.DataFrame(scores_df)
    test_scores = scores_combined

    confidence = 0.95

    # total number of documents is the number of ones in the binary array
    n = sum([np.sum(item) for item in prediction_dict.values()])

    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((test_scores * (1 - test_scores)) / n)

    ci_lower = (test_scores - ci_length).round(2)
    ci_upper = (test_scores + ci_length).round(2)

    precision_ci = ci_lower['precision'].astype(str) + '-' + ci_upper[
        'precision'].astype(str)
    recall_ci = ci_lower['recall'].astype(str) + '-' + ci_upper[
        'recall'].astype(str)
    f1_ci = ci_lower['F1'].astype(str) + '-' + ci_upper['F1'].astype(str)

    out = pd.DataFrame(scores_df)
    out = out.rename({0: 'value'}, axis=1)
    out['support'] = sum(
        [np.sum(item).astype(int) for item in gold_standard_dict.values()])
    if show_confidence_bounds:
        out['CI Precision'] = precision_ci
        out['CI Recall'] = recall_ci
        out['CI F1'] = f1_ci

    return out


def evaluation_report(gold_standard_json, prediction_json, round_num=2,
                      show_confidence_bounds=True):

    print(calculate_mean_scores(gold_standard_json, prediction_json,
                                show_confidence_bounds=show_confidence_bounds).round(round_num))

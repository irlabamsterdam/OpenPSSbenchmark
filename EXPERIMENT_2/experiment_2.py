"""
Because of the long time it takes to obtain the predictions for the datasets
We run the experiment 2 by using the predictions we have saved previously and
use these to calculate scores.
"""

import argparse
from metricutils import *


def main(arguments):
    if arguments.test_dataset == 'SAME_CORPUS':
        predictions_C1 = read_json('../resources/model_outputs/%s_C1/C1_C1/predictions.json' % arguments.modality)
        predictions_C2 =read_json('../resources/model_outputs/%s_C2/C2_C2/predictions.json'% arguments.modality)
    else:
        predictions_C1 = read_json('../resources/model_outputs/%s_C1/C1_C2/predictions.json' % arguments.modality)
        predictions_C2 =read_json('../resources/model_outputs/%s_C2/C2_C1/predictions.json' % arguments.modality)

    gold_standard_C1 = read_json('../resources/gold_standard/C1_test/gold_standard.json')
    gold_standard_C2 = read_json('../resources/gold_standard/C2_test/gold_standard.json')

    evaluation_report({**gold_standard_C1, **gold_standard_C2},
                      {**predictions_C1, **predictions_C2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['SAME_CORPUS', 'DIFFERENT_CORPUS'])
    parser.add_argument('--modality', choices=['WIED-TXT', 'WIED-IMG',
                                               'GUHA-TXT'])

    arguments = parser.parse_args()
    main(arguments)

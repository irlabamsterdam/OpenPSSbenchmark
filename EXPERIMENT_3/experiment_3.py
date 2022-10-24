"""
This script can be used to perform a full run of one of the fusion techniques
where we run it for all dataset configurations, read in the results
and calculate the combined scores.
"""

import argparse
from utils import *
from dataloading import *
from metricutils import *



def main(arguments):
    file_dict = {'EARLY': 'wiedemannearlyfusion.py',
                 'WIED-MM': 'wiedemannlatefusion.py',
                 'AUTOENCODER': 'autoencoder.py',
                 'LOGISTICREGRESSION': 'logisticregression.py'}

    for train_setting in ['C1', 'C2']:
        for test_setting in ['C1', 'C2']:
            print("SCORES FOR train-%s test-%s" % (train_setting, test_setting))
            os.system("python %s --train_dataset %s --test_dataset %s" % (file_dict[arguments.fusion_type],
                                                                   train_setting, test_setting))


    if arguments.test_dataset == 'SAME_CORPUS':
        predictions_C1 = read_json('../resources/experiment3_outputs/%s/C1_C1/predictions.json' % arguments.fusion_type)
        predictions_C2 =read_json('../resources/experiment3_outputs/%s/C2_C2/predictions.json'% arguments.fusion_type)
    else:
        predictions_C1 = read_json('../resources/experiment3_outputs/%s/C1_C2/predictions.json' % arguments.fusion_type)
        predictions_C2 =read_json('../resources/experiment3_outputs/%s/C2_C1/predictions.json' % arguments.fusion_type)

    gold_standard_C1 = read_json('../resources/gold_standard/C1_test/gold_standard.json')
    gold_standard_C2 = read_json('../resources/gold_standard/C2_test/gold_standard.json')

    print("SCORES FOR FINAL MODEL")
    evaluation_report({**gold_standard_C1, **gold_standard_C2},
                      {**predictions_C1, **predictions_C2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['SAME_CORPUS', 'DIFFERENT_CORPUS'])
    parser.add_argument('--fusion_type', choices=['LOGISTICREGRESSION',
                                                  'AUTOENCODER',
                                                  'WIED-MM',
                                                  'EARLY'])

    arguments = parser.parse_args()
    main(arguments)

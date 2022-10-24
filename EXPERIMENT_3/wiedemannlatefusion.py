import argparse

from utils import *
from metricutils import *


def combine_text_and_image_predictions(text_dict, image_dict, i=0.5, j=0.5):
    predictions = {}

    for stream_id in text_dict.keys():
        text_prob = np.array(text_dict[stream_id])
        visual_prob = np.array(image_dict[stream_id])

        text_out = np.vstack(
            [1 - text_prob, text_prob])  # probability from text model
        visual_out = np.vstack(
            [1 - visual_prob, visual_prob])  # probability from visual model

        combined_out = np.power(text_out, i) * np.power(visual_out, j)
        predictions[stream_id] = np.argmax(combined_out, axis=0).tolist()

    return predictions


def main(arguments):
    # The first step is to load in the train and test gold standard
    # As well as the train and output probabilities.
    experiment_data = load_data_for_experiment_3(arguments.train_dataset,
                                                 arguments.test_dataset,
                                                 fusion_type='LATE')

    # In this model we don't learn anything, so we just do our combination
    # on the test data
    test_data_text, test_data_image, test_labels = experiment_data['test_data_text'],\
                                                   experiment_data['test_data_image'],\
                                                   experiment_data['test_gold_standard']

    # Now that we have all the data we can set up the experiment.
    # Now we train the model and test it on the test data

    predictions = combine_text_and_image_predictions(test_data_text, test_data_image,
                                                     i=arguments.text_weight,
                                                     j=arguments.image_weight)

    with open('../resources/experiment3_outputs/WIED-MM/%s_%s/predictions.json' % (arguments.train_dataset,
                                                                                 arguments.test_dataset), 'w') as json_file:
        json.dump(predictions, json_file)

    evaluation_report(test_labels, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2'])
    parser.add_argument('--text_weight', type=float, default=0.5)
    parser.add_argument('--image_weight', type=float, default=0.5)
    arguments = parser.parse_args()
    main(arguments)
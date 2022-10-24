import argparse

from utils import *
from sklearn.linear_model import LogisticRegression


def predict(trained_model, image_dict, text_dict):
    predictions = {}
    for key, val in image_dict.items():
        image_vec = image_dict[key]
        text_vec = text_dict[key]
        combined_vec = np.vstack([image_vec, text_vec]).T
        predictions[key] = trained_model.predict(combined_vec)

    return predictions


def main(arguments):
    # The first step is to load in the train and test gold standard
    # As well as the train and output probabilities.
    experiment_data = load_data_for_experiment_3(arguments.train_dataset,
                                                 arguments.test_dataset,
                                                 fusion_type='LATE')

    train_data_text, train_data_image, train_labels = experiment_data['train_data_text'],\
                                                      experiment_data['train_data_image'],\
                                                      experiment_data['train_gold_standard']

    test_data_text, test_data_image, test_labels = experiment_data['test_data_text'],\
                                                   experiment_data['test_data_image'],\
                                                   experiment_data['test_gold_standard']

    # Now that we have all the data we can set up the experiment.
    # Now we train the model and test it on the test data
    train_X, train_y = prepare_training_data_experiment_3(train_data_image, train_data_text, train_labels)

    print('--training the logistic regression model--')
    logistic_model = LogisticRegression()
    logistic_model.fit(train_X, train_y)

    # Now we can make a prediction
    print("Now we perform the prediction")
    logistic_predictions = predict(logistic_model, test_data_image, test_data_text)
    evaluation_report(test_labels, logistic_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2_SAME', 'C1C2_DIFF'])

    arguments = parser.parse_args()
    main(arguments)
"""
In this file we will recreate the model that Pepijn used for the image
classification part of the thesis.
"""

import os
import argparse
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Local imports
from utils import *
from metricutils import *
from dataloading import *


class ImageModelWiedemann:
    def __init__(self, learning_rate=0.00001):
        # We use the VGG16 model pretrained on the imagenet corpus
        # As the basis of our network.
        model_vgg16 = VGG16(weights='imagenet', include_top=False,
                            input_shape=(300, 300, 3))

        # We don't want to train the first 13 layers of the VGG16 model
        # We will add our own tower to this later. It is common in the literature
        # To only freeze the first 4 of the 5 convolutional layers so that
        # the network can still learn to adjust some of the filters to specifics
        # of the dataset
        for l in model_vgg16.layers[:13]:
            l.trainable = False

        top_model = Flatten()(model_vgg16.output)
        drop1 = Dropout(0.5)(top_model)
        dense1 = Dense(512)(drop1)
        relu1 = LeakyReLU()(dense1)
        drop2 = Dropout(0.5)(relu1)
        dense2 = Dense(256)(drop2)
        relu2 = LeakyReLU()(dense2)
        # After the output of the model, we pass the output through
        # A final linear layer and a sigmoid to obtain values for prediction
        model_output = Dense(1, activation="sigmoid")(relu2)

        model = Model(model_vgg16.input, model_output)
        # Set up the optimzation steps as described in the original
        # wiedemann paper.
        model.compile(loss='binary_crossentropy', optimizer=Nadam(learning_rate=learning_rate),
                      metrics=['AUC'])
        self.model = model

    def train(self, train_data, num_epochs=20):
        self.model.fit(train_data, epochs=num_epochs)

    def predict(self, test_data):
        y_predict = self.model.predict(test_data, verbose=True)
        return y_predict


def prepare_df_for_model(dataframe, dataset):
    dataframe['png'] = dataframe.name + '-' + dataframe.page.astype(str) + '.png'
    if dataset == 'TOBACCO':
        dataframe['png'] = dataframe.name + '.png'
    dataframe['label'] = dataframe['label'].astype(str)

    return dataframe


def prepare_test_streams(test_subdataframe, png_folder,
                         batch_size):

    subtest_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input).flow_from_dataframe(
        dataframe=test_subdataframe,
        directory=png_folder,
        x_col='png',
        y_col='label',
        target_size=(300, 300),
        class_mode=None,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        validate_filenames=True,
    )

    return subtest_generator


def main(arguments):

    training_params = load_data_for_experiment_1(arguments.train_dataset,
                                                 arguments.test_dataset,
                                                 'WIED-IMG')

    train_dataframe = prepare_df_for_model(training_params['train_dataframe'], arguments.train_dataset)

    test_dataframe = prepare_df_for_model(training_params['test_dataframe'], arguments.test_dataset)
    pretrained_model_path = training_params['pretrained_model_path']

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input).flow_from_dataframe(
        dataframe=train_dataframe,
        directory=training_params['train_png_folder'],
        x_col='png',
        y_col='label',
        target_size=(300, 300),
        class_mode='binary',
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=42,
        validate_filenames=True)

    # We either want to train our own model and save it, or use a
    # Model we trained ourselves, and only run the prediction step.

    model = ImageModelWiedemann(learning_rate=arguments.learning_rate)
    if arguments.from_scratch:
        model.train(train_data=train_gen, num_epochs=arguments.num_epochs)
        model.model.save(pretrained_model_path)
    else:
        model.model = load_model(pretrained_model_path)


    stream_predictions = {}
    vector_outputs = {}

    intermediate_layer = model.model.get_layer("dense").output
    feature_extractor = Model(inputs=model.model.input, outputs=intermediate_layer)

    for doc_id, stream in test_dataframe.groupby('name'):

        test_data = prepare_test_streams(stream, training_params['test_png_folder'],
                                         arguments.batch_size)

        out = model.predict(test_data).squeeze()
        stream_prediction = np.round(out).astype(int).tolist()
        if isinstance(stream_prediction, int):
            stream_prediction = [stream_prediction]
        stream_predictions[doc_id] = stream_prediction

    test_dataframe['label'] = test_dataframe['label'].astype(int)

    gold_standard = get_ground_truth_from_dataframe(test_dataframe,
                                                    'label')

    evaluation_report(gold_standard, stream_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2_SAME', 'C1C2_DIFF'])
    arguments = parser.parse_args()
    main(arguments)

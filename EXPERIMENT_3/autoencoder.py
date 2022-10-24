import tqdm
import argparse
from keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Nadam

# Local imports
from utils import *
from metricutils import *


class FusionFeatureGenerator(Sequence):
    def __init__(self, vec_matrix, labels, batch_size=32,
                 preprocessor=None):
        self.vec_matrix = vec_matrix
        self.labels = labels
        self.indices = np.arange(len(self.vec_matrix))
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def __len__(self):
        return math.ceil(len(self.vec_matrix) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.vec_matrix[inds], self.labels[inds]
        if self.preprocessor:
            batch_x = self.preprocessor(batch_x)
        return batch_x, batch_y

class AutoEncoder:
    def __init__(self, input_shape, learning_rate: float = 0.002,
                 dropout: float=0.5, encoded_size: int= 100):
        model_input = Input(shape=(input_shape, ))
        intermediate = Dense(encoded_size)(model_input)
        intermediate = LeakyReLU()(intermediate)
        intermediate = Dropout(dropout)(intermediate)
        model_output = Dense(input_shape)(intermediate)
        model = Model(model_input, model_output)
        model.compile(loss='mse', optimizer=Nadam(learning_rate=learning_rate),
                               metrics=['cosine_similarity'])
        self.model = model

    def train(self, train_data, train_labels, batch_size: int = 512,
              num_epochs: int = 25):
        self.model.fit(FusionFeatureGenerator(train_data, train_labels,
                                              batch_size=batch_size),
                       epochs=num_epochs)

class SimpleClassifier:
    def __init__(self, input_shape: int, learning_rate: float = 0.002,
                 dropout: float = 0.5):
        model_input = Input(shape=(input_shape,))
        combined = Dense(200)(model_input)
        combined = LeakyReLU()(combined)
        combined = Dropout(dropout)(combined)
        model_output = Dense(1, activation='sigmoid')(combined)
        combined_model = Model(model_input, model_output)
        combined_model.compile(loss='binary_crossentropy', optimizer=Nadam(learning_rate=learning_rate),
                               metrics=['accuracy'])
        self.model = combined_model

    def train(self, encoder_model, train_data, train_labels, batch_size: int = 512,
              num_epochs: int = 25):
        self.model.fit(FusionFeatureGenerator(train_data, train_labels,
                                              batch_size=batch_size,
                                              preprocessor=encoder_model),
                       epochs=num_epochs)


def main(arguments):
    # The first step is to load in the train and test gold standard
    # As well as the train and output probabilities.
    experiment_data = load_data_for_experiment_3(arguments.train_dataset,
                                                 arguments.test_dataset,
                                                 fusion_type='EARLY')

    # In this model we don't learn anything, so we just do our combination
    # on the test data

    train_data_text, train_data_image, train_labels = experiment_data['train_data_text'],\
                                                      experiment_data['train_data_image'],\
                                                      experiment_data['train_gold_standard']

    test_data_text, test_data_image, test_labels = experiment_data['test_data_text'],\
                                                   experiment_data['test_data_image'],\
                                                   experiment_data['test_gold_standard']

    # Now that we have all the data we can set up the experiment.
    # Now we train the model and test it on the test data

    vecs_train, labels_train = combined_vectors_with_gold_standard(
        train_data_text,
        train_data_image,
        train_labels)

    encoder_model = AutoEncoder(input_shape=vecs_train.shape[1],
                                learning_rate=arguments.learning_rate,
                                dropout=0.5,
                                encoded_size=arguments.compression_size)

    encoder_model.train(vecs_train, labels_train, num_epochs=arguments.num_epochs,
                        batch_size=arguments.batch_size)

    # We want the middle layer of the network as this represents our compressed
    # vector
    intermediate_layer = encoder_model.model.get_layer("dense").output
    feature_extractor = Model(inputs=encoder_model.model.input,
                              outputs=intermediate_layer)

    fusion_model = SimpleClassifier(input_shape=arguments.compression_size,
                                    learning_rate=arguments.learning_rate,
                                    dropout=0.5)

    fusion_model.train(feature_extractor, vecs_train, labels_train,
                       num_epochs=arguments.num_epochs,
                       batch_size=arguments.batch_size)

    # This is step 1, now we retrieve the embeddings from this trained model
    # and use them to train the classifier model, that model will provide our
    # final predictions

    all_predictions = {}

    for stream in tqdm.tqdm(test_labels.keys()):
        test_vecs_image = test_data_image[stream]
        test_vecs_text = test_data_text[stream]

        combined_test_vecs = np.hstack([test_vecs_text, test_vecs_image])
        encoded_vecs = feature_extractor(combined_test_vecs)

        outputs = fusion_model.model.predict(encoded_vecs).round().squeeze()
        all_predictions[stream] = outputs.tolist()

    evaluation_report(test_labels, all_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2_SAME', 'C1C2_DIFF'])
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--compression_size', type=int, default=100)
    arguments = parser.parse_args()
    main(arguments)








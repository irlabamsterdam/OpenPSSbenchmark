"""
In this script the code for the Text CNN model from Wiedemann et al is
given, with the code being nearly identical to the original model,
with the exception of the word vectors, as we are now loading vectors for the
Dutch Language.
"""

import re
import math
import argparse
from tqdm import tqdm
from typing import List
# Here we define the function so we can load the train and testset dataframes


from gensim.models.fasttext import load_facebook_model

from tensorflow.keras.utils import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

# Local imports
from metricutils import *
from utils import *

# Due to Tensorflow version constraints / incompatibilities with Fasttext
# We use gensim to load the word vectors as a workaround.
ft = load_facebook_model("../resources/fasttext_vectors/cc.nl.300.bin")
print('---Fasttext Vectors has been loaded---')


def get_data_instances(df) -> List[List[str]]:
    """
    :param df: Dataframe with the data for the classification model
    The dataframe should contain at least two columns, with the names
    'label' and 'text', containing label and text for the model respectively.
    :return: A list of lists, where each element in the outer list represents
    one datapoint, and this list itself contains two elements, the label and
    the text of that datapoint.
    """
    data_instances = []
    for index, row in df.iterrows():
        data_instances.append([row['label'], row['text']])
    return data_instances


def simple_tokenizer(textline: str) -> List[str]:
    """
    :param textline: String containin the text of one page
    :return: A list representing the tokenized sentence.
    """
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(
        textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return words


class TextFeatureGenerator(Sequence):
    """
    This class is used to generate batches of data as input to the text cnn
    model.
    """
    def __init__(self, text_data, batch_size=32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.sequence_length = 150
        self.embedding_dims = 300

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):
        """
        :param inds: List of indices into the self.text_data array
        :return: two arrays with the word embeddings and teh labels of ech
        page respectively.
        """
        word_embeddings = []
        output_labels = []

        for index in inds:
            # Retrieve a word embedding for each word in the page sequentially
            word_embeddings.append(
                self.text_to_embedding(self.text_data[index][1]))
            output_labels.append(self.text_data[index][0])

        return np.array(word_embeddings), np.array(output_labels)

    def text_to_embedding(self, textsequence):
        temp_word = []

        # tokenize
        sentence = simple_tokenizer(textsequence)
        # trim to max sequence length
        if len(sentence) > self.sequence_length:
            half_idx = int(self.sequence_length / 2)
            tmp_sentence = sentence[:half_idx]
            tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
            sentence = tmp_sentence

        # padding
        words_to_pad = self.sequence_length - len(sentence)

        for i in range(words_to_pad):
            sentence.append('PADDING_TOKEN')

        # create data input for words
        for w_i, word in enumerate(sentence):

            if word == 'PADDING_TOKEN':
                word_vector = [0] * self.embedding_dims
            else:
                word_vector = ft.wv[word.lower()]

            temp_word.append(word_vector)

        return temp_word


class TextModelWiedemann:
    def __init__(self, nb_embedding_dims=300, nb_sequence_length=150):

        # Use the three filters as described in the original Text CNN
        # paper from KIM
        filter_sizes = (3, 4, 5)

        model_input_tp = Input(shape=(nb_sequence_length, nb_embedding_dims))

        gru_block_tp = Bidirectional(
            GRU(128, dropout=0.5, return_sequences=True))(
            model_input_tp)

        conv_blocks_tp = []

        for sz in filter_sizes:
            conv = Conv1D(
                filters=200,
                kernel_size=sz,
                padding="same",
                strides=1)(gru_block_tp)
            conv = LeakyReLU()(conv)
            conv = GlobalMaxPooling1D()(conv)
            conv = Dropout(0.5)(conv)
            conv_blocks_tp.append(conv)

        # Concatenate the output of the three separate filters sizes used
        model_concatenated_tp = concatenate(conv_blocks_tp)
        model_concatenated_tp = Dense(128)(model_concatenated_tp)
        model_concatenated_tp = LeakyReLU()(model_concatenated_tp)

        # concat both + another dense
        model_concatenated_tp = Dense(256)(model_concatenated_tp)
        model_concatenated_tp = LeakyReLU()(model_concatenated_tp)

        model_output = Dense(1, activation="sigmoid")(model_concatenated_tp)

        # combine final model
        model = Model(model_input_tp, model_output)
        model.compile(loss='binary_crossentropy', optimizer='nadam',
                      metrics=['accuracy'])

        self.model = model

    def train(self, train_data, batch_size, num_epochs):
        # Here we write a very simple training loop
        self.model.fit(TextFeatureGenerator(train_data, batch_size=batch_size),
                       epochs=num_epochs)

    def predict(self, test_dataframe, batch_size: int, feature_extractor):
        all_stream_predictions = {}
        raw_predictions = {}
        all_vectors = {}

        for name, sub_df in tqdm(test_dataframe.groupby("name")):
            # Predictions will return the final activation of the model after
            # the sigmoid.
            predictions = self.model.predict(
                TextFeatureGenerator(get_data_instances(sub_df),
                                     batch_size=batch_size)).squeeze()

            if predictions.size == 1:
                predictions = np.array([predictions])

            vectors = feature_extractor.predict(TextFeatureGenerator(get_data_instances(sub_df),
                                                                     batch_size=batch_size))

            all_vectors[name] = vectors
            # Sigmoid is always between 0 and 1, so rounding will give us
            # the right predictions
            final_predictions = predictions.round().astype(int).tolist()
            all_stream_predictions[name] = final_predictions
            raw_predictions[name] = predictions.tolist()
        return all_stream_predictions, raw_predictions, all_vectors


def main(arguments):
    training_params = load_data_for_experiment_1(arguments.train_dataset,
                                                 arguments.test_dataset,
                                               'WIED-TXT')
    train_dataframe = training_params['train_dataframe']
    test_dataframe = training_params['test_dataframe']
    pretrained_model_path = training_params['pretrained_model_path']

    # Our first step here is to set up the model from its class
    if arguments.from_scratch:
        text_model = TextModelWiedemann()
    else:
        text_model = TextModelWiedemann()
        text_model.model = load_model(pretrained_model_path)
    text_model.ft = ft
    print('Model has been loaded')

    # Now we also have to load the training and test datasets
    gold_standard_dict = get_ground_truth_from_dataframe(test_dataframe,
                                                         col='label')

    train_instances = get_data_instances(train_dataframe)

    if arguments.from_scratch:
        text_model.train(train_instances, batch_size=arguments.batch_size,
                         num_epochs=arguments.num_epochs)
        text_model.model.save(arguments.save_path)

    # This gives us the layer to get outputs with 256 neurons, as in
    # the Wiedemann et al paper.ß
    intermediate_layer = text_model.model.get_layer('dense_1').output
    feature_extractor = Model(inputs=text_model.model.input,
                              outputs=intermediate_layer)

    if arguments.use_existing_predictions:
        prediction_dict = get_existing_predictions(arguments.train_dataset,
                                                   arguments.test_dataset,
                                                   'WIED-TXT')['predictions']
    else:
        prediction_dict, output_vectors, _ = text_model.predict(test_dataframe,
                                                             batch_size=arguments.batch_size,
                                                             feature_extractor=feature_extractor)

        np.save('../resources/model_outputs/WIED-TXT_%s/%s_%s' % (arguments.train_dataset,
                                                                  arguments.train_dataset,
                                                                  arguments.test_dataset), output_vectors)


        with open('../resources/model_outputs/WIED-TXT_%s/%s_%s/predictions.json' % (arguments.train_dataset,
                                                                                     arguments.train_dataset,
                                                                                     arguments.test_dataset), 'w') as json_file:
            json.dump(prediction_dict, json_file)

    # Now also save the predictions

    # We use the evaluation functionality from the metricutils file.
    evaluation_report(gold_standard_dict, prediction_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--use_existing_predictions', default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2'])
    arguments = parser.parse_args()
    main(arguments)

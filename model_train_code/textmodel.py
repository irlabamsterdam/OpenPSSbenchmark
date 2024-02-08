"""
This file implements a recreation of the Textual model used by Pepijn.
We have reimplemented the code so that it is compatible with the other models
we have in this directory, this making it much easier to later perform the
model fusion experiments.
"""

import re
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torchtext.vocab import FastText
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer

import fasttext

embed_model = fasttext.load_model("/ivi/ilps/personal/rheusde/WOOIR/models/cc.nl.300.bin")

# local imports
import metricutils
from dataloading import load_text_dataframe


class WOBDatasetText(torch.utils.data.Dataset):
    def __init__(self, dataframe_path: str):

        self.dataframe_path = dataframe_path
        data_df = load_text_dataframe(dataframe_path)

        self.labels = data_df['label'].tolist()
        self.text = data_df['text'].tolist()

    def _tokenize(self, sentence):
        textline = re.sub(r'http\S+', 'URL', sentence)
        words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(
            textline.strip())
        words = [w.strip() for w in words if w.strip() != '']
        if not words:
            return [' ']
        return (words)

    def _embed_text(self, page: List):
        vector_batch = torch.zeros((1, 150, 300))
        tok_sentence = self._tokenize(page)
        if len(tok_sentence) > 150:
            tok_sentence = tok_sentence[:75] + tok_sentence[-75:]
        vectors = torch.from_numpy(np.array([embed_model.get_word_vector(word.lower()) for word in tok_sentence]))
        vector_batch[0, :vectors.shape[0], :] = vectors
        return vector_batch, len(tok_sentence)

    def __getitem__(self, idx):
        datapoint_label = self.labels[idx]
        datapoint_text, length = self._embed_text(self.text[idx])

        return (datapoint_text, length), datapoint_label

    def __len__(self):
        return len(self.labels)


class TextModel(LightningModule):
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def __init__(self, learning_rate: float):
        super().__init__()

        self.gru = nn.GRU(input_size=300, hidden_size=128, bidirectional=True,
                          batch_first=True)

        self.conv_kernel_size_3 = nn.Conv1d(in_channels=150, out_channels=200,
                                            kernel_size=3, padding='same')

        self.conv_kernel_size_4 = nn.Conv1d(in_channels=150, out_channels=200,
                                            kernel_size=4, padding='same')

        self.conv_kernel_size_5 = nn.Conv1d(in_channels=150, out_channels=200,
                                            kernel_size=5, padding='same')

        self.max_pool_3 = nn.MaxPool1d(kernel_size=3)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=4)
        self.max_pool_5 = nn.MaxPool1d(kernel_size=5)

        self.final_layer = nn.Linear(40000, 128)

        self.classifier = nn.Linear(128, 1)

        # Define the loss function
        self.loss_function = nn.BCEWithLogitsLoss()

        self.learning_rate = learning_rate

        self.init_weights()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h_0 = torch.zeros(2, x[0].shape[0], 128)

        torch.nn.init.xavier_normal_(h_0)
        inputs, lengths = x
        x = nn.utils.rnn.pack_padded_sequence(inputs.squeeze(), lengths.cpu(),
                                              batch_first=True,
                                              enforce_sorted=False)
        x, states = self.gru(x, h_0)
        conv_input, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True,
                                                         total_length=150)

        # I think I should apply dropout here now
        conv_input = F.dropout(conv_input, p=0.5)

        conv_3 = self.conv_kernel_size_3(conv_input)
        conv_3 = F.leaky_relu(conv_3)
        conv_3 = self.max_pool_3(conv_3)
        conv_3 = F.dropout(conv_3, p=0.5)

        conv_4 = self.conv_kernel_size_4(conv_input)
        conv_4 = F.leaky_relu(conv_4)
        conv_4 = self.max_pool_4(conv_4)
        conv_4 = F.dropout(conv_4, p=0.5)

        conv_5 = self.conv_kernel_size_5(conv_input)
        conv_5 = F.leaky_relu(conv_5)
        conv_5 = self.max_pool_5(conv_5)
        conv_5 = F.dropout(conv_5, p=0.5)

        conv_output = torch.cat([conv_3, conv_4, conv_5], 2)
        conv_output = conv_output.reshape(conv_output.size(0), -1)

        final_out = self.final_layer(conv_output)
        final_out = F.leaky_relu(final_out)

        final_out = self.classifier(final_out)

        return final_out.squeeze()

    def training_step(self, batch):
        x, y = batch
        prediction = self(x)
        loss = self.loss_function(prediction.float(), y.float())
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """

        This function implements one test step and is called when running
        'trainer.test()'. It returns the predictions of the model and the ground
        truth for one input batch.

        :param batch: torch.Tensor of shape (N_SAMPLES, N_CHANNELS, WIDTH,
        HEIGHT)
        :param batch_idx: integer specifying the unique numerical ID of the
        batch
        :return: float specifying the loss of the model for the given input
        batch
        """
        x, y = batch
        prediction = self(x)

        # We don't have a sigmoid in our model by default because we use
        # BCELosswithLogits, so for the test predictions we have to add this
        # back in.
        model_output = self.sigmoid(prediction)
        return model_output, y

    def test_epoch_end(self, outputs):
        """
        This function is called at the end of the test epoch and will combine
        the predictions of the model on the individual batches to calculate a
        final score of the model indicating its performance.
        :param outputs: list of outputs of the model prediction on each input
        batch. the length of the list is equal to the number of batches,
        and each sublist contains two elements, the first one being
        predictions of the model, the second one being the actual ground truth
        labels.
        """
        predictions = []
        labels = []

        # Go through all batches and add both the predictions and ground truth
        # to the large lists that will be used for the final score calculation
        for batch in outputs:
            batch_predictions, batch_labels = batch
            predictions.extend(batch_predictions.detach().round().int().tolist())
            labels.extend(batch_labels.detach().int().tolist())

        df = pd.DataFrame({'label': labels, 'prediction': predictions})
        df.to_csv('outputs.csv', index=False)

    def configure_optimizers(self):
        """
        This function configures that optimizers used during model training.
        The model follows that approach used in the Wiedemann paper,
        and uses the NAdam loss function.
        :return:
        """
        return torch.optim.NAdam(self.gru.parameters(), lr=self.learning_rate,
                                 eps=1e-07, weight_decay=0, momentum_decay=0)


def main(arguments):
    if arguments.do_train:
        wob_text_model = TextModel(learning_rate=arguments.learning_rate)
    else:
        wob_text_model = TextModel.load_from_checkpoint(arguments.saved_model_path)
        wob_text_model.eval()
    # First we define the train and test dataloaders
    train_dataset = WOBDatasetText(arguments.train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=arguments.batch_size,
                                  shuffle=True, num_workers=2)

    test_dataset = WOBDatasetText(arguments.test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch_size,
                                 shuffle=False, num_workers=2)

    dataframe_test = pd.read_csv(arguments.test_path)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="loss")

    if args.do_train:
        trainer = Trainer(max_epochs=arguments.number_of_epochs,
                          callbacks=[checkpoint_callback], gpus=1)
        # trainer = Trainer(max_epochs=arguments.number_of_epochs,
        #                   callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(max_epochs=1, gpus=0,
                          limit_train_batches=0, limit_val_batches=0,
                          callbacks=[checkpoint_callback])

    trainer.fit(wob_text_model, train_dataloaders=train_dataloader)
    if args.do_train:
        trainer.test(wob_text_model, test_dataloader,
                     ckpt_path='best')
    else:
        trainer.test(wob_text_model, test_dataloader,
                     ckpt_path=arguments.saved_model_path)

    pred_df = pd.read_csv('outputs.csv')
    pred_df['name'] = dataframe_test['name'].tolist()

    # Now we get the dicts and compare
    gold = metricutils.get_ground_truth_from_dataframe(pred_df, 'label')
    output = metricutils.get_ground_truth_from_dataframe(pred_df, 'prediction')

    metricutils.evaluation_report(gold, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--number_of_epochs', type=int, default=5)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--saved_model_path', type=str, required=False)

    args = parser.parse_args()

    main(args)



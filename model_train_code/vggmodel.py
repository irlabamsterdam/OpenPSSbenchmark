"""
Here I try to implement the VGG model that Pepijn used, but then using
Pytorch lightning, so that hopefully speeds up the process significantly.
"""


import torch
import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Local imports
import metricutils
from dataloading import WOBDatasetImage, load_text_dataframe

#import utils.metricutils
#from utils.dataloading import WOBDatasetImage


class ImageModelWOB(LightningModule):
    def __init__(self, learning_rate: float = 0.00001):
        """
        This class implements the VGG16 model used for the classification
        of document boundaries for WOB documents. This class can also be used
        to extract image embeddings from the VGG16 model after training.
        :param learning_rate: float specifying the learning rate of the VGG16
        model.
        """
        super().__init__()

        # First initialize the VGG network
        backbone_tmp = vgg16(weights=VGG16_Weights.IMAGENET1K_V1,
                                progress=True)

        # Keep the original VGG network, but discard the classification tower
        # As we are going to use our own version for this
        frozen_layers = list(backbone_tmp.children())[:-1]
        self.backbone = nn.Sequential(*frozen_layers)

        # Initialize the classification tower as used by Wiedemann et al.
        # in their paper
        self.classifier = nn.Sequential(nn.Flatten(),
                                   nn.Dropout(0.5),
                                   nn.Linear(25088, 512),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 1))

        # Because we are doing a binary classification task, we will use the
        # binary cross entropy loss. For numerical stability, we use the
        # version of the function that uses logits, and this version also
        # includes a sigmoid layer.
        self.loss_function = nn.BCEWithLogitsLoss()

        # set a learning rate, this can be left as is to use the default value
        # from the Wiedemann paper
        self.learning_rate = learning_rate

        # Set up a sigmoid for the test steps of the model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image: torch.Tensor):
        """
        This function defines with happens when a batch of input images is fed
        into the model. It first gets passed through the base vgg16 model, and
        then gets classified into either 0 or 1 using the custom classification
        layer we defined in __ini__

        :param input_image: Batch of inputs with shape (BATCH, N_CHANNELS,
        WIDTH, HEIGHT)
        :return: raw output of the model after passing it through the
        base VGG16 and the classification layer.
        """

        # We don't train the base model so we set the backbone to eval mode
        # to prevent the gradients from flowing through this part of the network
        self.backbone.eval()
        with torch.no_grad():
            input_representation = self.backbone(input_image).flatten(1)

        x = self.classifier(input_representation)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        """
        This method implements a single training step for the model,
        which consists of running the input through the models forward function
        calculating the loss of the model with the BCELOSSwithLogits function
        and returning this loss for the optimizer.

        :param batch: torch.Tensor of shape (N_SAMPLES, N_CHANNELS, WIDTH,
        HEIGHT)
        :param batch_idx: integer specifying the unique numerical ID of the
        batch
        :return: float specifying the loss of the model for the given input
        batch
        """
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
        return torch.optim.NAdam(self.classifier.parameters(), lr=self.learning_rate,
                                 eps=1e-07, weight_decay=0, momentum_decay=0)


def main(arguments):

    if arguments.do_train:
        wob_image_model = ImageModelWOB(learning_rate=arguments.learning_rate)
    else:
        wob_image_model = ImageModelWOB.load_from_checkpoint(arguments.saved_model_path)
        wob_image_model.eval()

    train_dataset = WOBDatasetImage(arguments.train_path,
                                    arguments.png_folder_train)

    test_dataset = WOBDatasetImage(arguments.test_path,
                                   arguments.png_folder_test)

    dataframe_test = pd.read_csv(arguments.test_path)

    train_dataloader = DataLoader(train_dataset, batch_size=arguments.batch_size,
                                  shuffle=True, num_workers=8)

    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch_size,
                                 shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="loss")

    if args.do_train:
        trainer = Trainer(max_epochs=arguments.number_of_epochs, gpus=1,
                          callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(max_epochs=1, gpus=1,
                          limit_train_batches=0, limit_val_batches=0,
                          callbacks=[checkpoint_callback])

    trainer.fit(wob_image_model, train_dataloaders=train_dataloader)
    if args.do_train:
        trainer.test(wob_image_model, test_dataloader,
                     ckpt_path='best')
    else:
        trainer.test(wob_image_model, test_dataloader,
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
    parser.add_argument('--png_folder_train', type=str, required=True)
    parser.add_argument('--png_folder_test', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--number_of_epochs', type=int, default=20)
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--saved_model_path', type=str, required=False)

    args = parser.parse_args()
    main(args)

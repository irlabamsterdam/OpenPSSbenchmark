"""
This file contains a PyTorch style dataloader that can be used to load in the
data from different WOB data sources, for easier use with the different models
that are used in this project.

Because we have two modalities for each page (both visual and textual) we
construct the dataset in such a way that we can use one dataloader for
the whole dataset, and just select what information we want to use based on
which model we are training.

For some of the models that we are using we need to specify a specific
pipeline for the preprocessing of the image, because we need to do some
resampling / rescaling, such as for the VGG16 model.

Although the exact structure of the dataset does not have to specified
in advance (train, val and test split) we will require this for the first versions
of the algorithms, just so that we know for sure that we actually do the
proper things for the first two corpora. Later we can then make a more
complicated version that will allow us a bit more freedom in how we interact
with the datasets.

"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Local imports
import metricutils


def load_text_dataframe(dataframe_path: str, nan_fill_value: str = ''):
    """
    In this method we load in the csv text OCR dataset.
    We do this in a separate method because the dataset requires
    some preprocessing to make sure that the data is loading in properly
    and that we can actually combine it with the images that we get from
    the image loading module.

    :param nan_fill_value: string specifying what value to use if a value
    is missing in the text entry of a page.
    :param dataframe_path: string specifying the path to the dataframe
    that contains the text of the pages and the gold standard.
    :return:
    """
    ocr_dataframe = pd.read_csv(dataframe_path)

    # As in principle these pages could be unordered, we want to make sure
    # stream is ordered in ascending order by the page number so that it
    # lines up with the gold standard data.

    ocr_dataframe['page'] = ocr_dataframe['page'].astype(int)
    # sorting by name because this is just the name of the stream and this
    # way we can sort each stream on page number properly.
    ocr_dataframe = ocr_dataframe.sort_values(by=['name', 'page'])

    ocr_dataframe.reset_index(inplace=True, drop=True)

    # Fill any nan values in the text with the value specified
    # in 'nan_fill_value'
    ocr_dataframe.text.fillna(nan_fill_value, inplace=True)

    return ocr_dataframe


class WOBDatasetText(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class WOBDatasetImage(torch.utils.data.Dataset):
    """
    TODO: rewrite this docstring so that it reflects that we can load
    images and text both separately and combined now.
    """
    def __init__(self, dataframe_path: str, png_folder: str = None,
                 preload_images: bool = False):

        self.dataframe_path = dataframe_path
        self.png_folder = png_folder
        self.preload_images = preload_images

        data_df = load_text_dataframe(dataframe_path)

        self.labels = data_df['label']

        image_names = data_df['name'] + '-' + data_df['page'].astype(str) + '.png'
        self.images = self.png_folder + os.sep + image_names
        self.image_transform = VGG16_Weights.IMAGENET1K_V1.transforms()
        if self.preload_images:
            self.images = [self._preprocess_image(image_path) for image_path in self.images]

    def _preprocess_image(self, image_path: str):
        image = Image.open(image_path)
        processed_image = self.image_transform(image)

        return processed_image

    def __getitem__(self, idx):
        datapoint_label = self.labels[idx]
        if self.preload_images:
            datapoint_image = self.images[idx]
        else:
            datapoint_image = self._preprocess_image(self.images[idx])

        return datapoint_image, datapoint_label

    def __len__(self):
        return len(self.labels)



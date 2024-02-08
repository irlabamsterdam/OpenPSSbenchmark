import os
import PIL
import argparse
import tensorflow
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


class ImageModelWiedemann:
    def __init__(self):

        # We use the VGG16 model pretrained on the imagenet corpus
        # As the basis of our network.
        model= VGG16(weights='imagenet', include_top=True,
                     input_shape=(224, 224, 3))

        # Include the top layer, but just get the vectors at the 'layer'

        for l in model.layers:
            l.trainable = False

        # After the output of the model, we pass the output through
        # A final linear layer and a sigmoid to obtain values for prediction

        self.intermediate_activation = Model(inputs=model.input,
                                             outputs=model.get_layer('fc1').output)


def prepare_df_for_model(dataframe):
    dataframe['png'] = dataframe.name + '-' + dataframe.page.astype(str) + '.png'
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
        target_size=(224, 224),
        class_mode=None,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        validate_filenames=True,
    )

    return subtest_generator


def main(args):

    test_dataframe = prepare_df_for_model(pd.read_csv(args.test_dataframe))

    # We either want to train our own model and save it, or use a
    # Model we trained ourselves, and only run the prediction step.

    model = ImageModelWiedemann()
    vector_outputs = {}

    for doc_id, stream in test_dataframe.groupby('name'):
        stream['page'] = stream['page'].astype(int)
        sorted_stream = stream.sort_values(by='page')

        test_data = prepare_test_streams(sorted_stream, args.test_png_folder,
                                         args.batch_size)

        vectors = model.intermediate_activation.predict(test_data)
        print(vectors.shape)
        vector_outputs[doc_id] = vectors

    print("Done with the feature extraction!")
    np.save(os.path.join(args.save_path, 'pretrained_vectors.npy'),
            vector_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataframe', type=str, required=True)
    parser.add_argument('--test_png_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_path', type=str)

    arguments = parser.parse_args()
    main(arguments)

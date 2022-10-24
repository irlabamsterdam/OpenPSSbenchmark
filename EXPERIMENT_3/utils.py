"""
This script contains some general function that are used across the different
experiments, such as the saving of the predictions of the different models in a
uniform manner.
"""

import math
from tensorflow.keras.utils import *

# Local imports
from dataloading import *
from metricutils import *

def combined_vectors_with_gold_standard(vector_dict_text, vector_dict_image, gold_standard_dict):
    labels = []
    vectors_text = []
    vectors_image = []

    for key in vector_dict_text.keys():
        labels.extend(gold_standard_dict[key])
        vectors_text.append(vector_dict_text[key])
        vectors_image.append(vector_dict_image[key])

    vector_matrix_text = np.concatenate(vectors_text)
    vector_matrix_image = np.concatenate(vectors_image)
    combined_vectors = np.concatenate([vector_matrix_text, vector_matrix_image],
                                      axis=1)
    return combined_vectors, np.array(labels)



def prepare_training_data_experiment_3(image_dict, text_dict, gold_standard_dict):
    labels = []
    training_vectors = []
    for key, val in image_dict.items():
        labels.extend(gold_standard_dict[key])
        image_vec = image_dict[key]
        text_vec = text_dict[key]
        combined_vec = np.vstack([image_vec, text_vec]).T
        training_vectors.append(combined_vec)
    return np.vstack(training_vectors), np.array(labels)


def load_vector_data(file_path: str):
    vectors = np.load(file_path, allow_pickle=True)
    output_dict = vectors[()]
    return output_dict

# Apart from running the models on separate datasets, we also have the
# option to run the robustness experiments on the combined datasets
# which we can then use to replicate the results from the paper.


def load_data_for_experiment_3(train_dataset: str, test_dataset: str,
                               fusion_type: str):
    assert train_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid train dataset string"
    assert test_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid test dataset string"
    assert fusion_type in ['EARLY', 'LATE'], "Please select a valid fusion type"

    # Here we have to get the train and test gold standard
    # as well as the train and test vectors, either the full vectors
    # or the output probabilities, depending on the fusion type.
    train_gold_standard = read_json('../resources/gold_standard/%s_train/gold_standard.json' % train_dataset)
    test_gold_standard =read_json('../resources/gold_standard/%s_test/gold_standard.json' % test_dataset)

    if fusion_type == 'LATE':
        train_data_text = read_json('../resources/model_outputs/WIED-TXT_%s/train/raw_scores.json' % train_dataset)
        train_data_image = read_json('../resources/model_outputs/WIED-IMG_%s/train/raw_scores.json' % train_dataset)

        test_data_text = read_json('../resources/model_outputs/WIED-TXT_%s/%s_%s/raw_scores.json' % (train_dataset, train_dataset, test_dataset))
        test_data_image = read_json('../resources/model_outputs/WIED-IMG_%s/%s_%s/raw_scores.json' % (train_dataset, train_dataset, test_dataset))
    # Else we use early fusion
    else:
        train_data_text = load_vector_data('../resources/model_outputs/WIED-TXT_%s/train/raw_vectors.npy' % train_dataset)
        train_data_image = load_vector_data('../resources/model_outputs/WIED-IMG_%s/train/raw_vectors.npy' % train_dataset)

        test_data_text = load_vector_data('../resources/model_outputs/WIED-TXT_%s/%s_%s/raw_vectors.npy' % (train_dataset, train_dataset, test_dataset))
        test_data_image = load_vector_data('../resources/model_outputs/WIED-IMG_%s/%s_%s/raw_vectors.npy' % (train_dataset, train_dataset, test_dataset))

    return {'train_gold_standard': train_gold_standard,
            'test_gold_standard': test_gold_standard,
            'train_data_text': train_data_text,
            'train_data_image': train_data_image,
            'test_data_text': test_data_text,
            'test_data_image': test_data_image}



def load_data_for_experiment_1(train_dataset: str, test_dataset: str,
                               modality: str):
    assert train_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid train dataset string"
    assert test_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid test ataset string"
    assert modality in ['WIED-TXT', 'WIED-IMG', 'GUHA-TXT'],\
        "Please select a valid modality"

    train_dataframe = load_text_dataframe(os.path.join('../resources/dataframes', train_dataset,
                                                       'train.csv'))
    test_dataframe = load_text_dataframe(os.path.join('../resources/dataframes', test_dataset,
                                                      'test.csv'))
    pretrained_model = os.path.join('../resources/trained_models',
                                    '%s_%s' % (modality, train_dataset))

    return {'train_dataframe': train_dataframe, 'test_dataframe': test_dataframe,
            'pretrained_model_path': pretrained_model}


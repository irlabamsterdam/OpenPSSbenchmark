"""
This script contains some general function that are used across the different
experiments, such as the saving of the predictions of the different models in a
uniform manner.
"""

# Local imports
from dataloading import *
from metricutils import *


def get_existing_predictions(train_corpus: str, test_corpus: str, modality: str):
    assert train_corpus in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid train dataset string"
    assert test_corpus in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid test ataset string"
    assert modality in ['WIED-TXT', 'WIED-IMG', 'GUHA-TXT'],\
        "Please select a valid modality"

    predictions = read_json('../model_outputs/%s_%s/%s_%s/predictions.json' % (modality,train_corpus, train_corpus, test_corpus))

    gold_standard = read_json('../gold_standard/%s_test/gold_standard.json'% test_corpus)

    return {'predictions': predictions, 'gold_standard': gold_standard}



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


def load_data_for_experiment_1(train_dataset: str, test_dataset: str,
                               modality: str):
    assert train_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid train dataset string"
    assert test_dataset in ['C1', 'C2', 'TOBACCO'],\
        "Please select a valid test ataset string"
    assert modality in ['WIED-TXT', 'WIED-IMG', 'GUHA-TXT'],\
        "Please select a valid modality"

    train_dataframe = load_text_dataframe(os.path.join('../dataframes', train_dataset,
                                                       'train.csv'))
    test_dataframe = load_text_dataframe(os.path.join('../dataframes', test_dataset,
                                                      'test.csv'))
    pretrained_model = os.path.join('../trained_models',
                                    '%s_%s' % (modality, train_dataset))
    if modality == 'WIED-IMG':
        train_pngs = '../%s_images/train' % train_dataset
        test_pngs = '../%s_images/test' % test_dataset
        return {'train_dataframe': train_dataframe, 'test_dataframe': test_dataframe,
                'pretrained_model_path': pretrained_model,
                'train_png_folder': train_pngs,
                'test_png_folder': test_pngs}

    return {'train_dataframe': train_dataframe, 'test_dataframe': test_dataframe,
            'pretrained_model_path': pretrained_model}


def save_model_predictions(gold_standard_dict: dict, predictions_dict: dict,
                           output_folder: str,
                           output_vectors=None,
                           probabilities:dict = None):

    if not os.path.exists(output_folder):
        # Create a new directory because it does not exist
        os.makedirs(output_folder)

    np.save(os.path.join(output_folder, 'raw_vecs.npy'), output_vectors)

    with open(os.path.join(output_folder, 'predictions.json'), 'w') as file:
        json.dump(predictions_dict, file)

    with open(os.path.join(output_folder, 'gold_standard.json'), 'w') as file:
        json.dump(gold_standard_dict, file)

    with open(os.path.join(output_folder, 'raw_scores.json'), 'w') as file:
        json.dump(probabilities, file)

    return None
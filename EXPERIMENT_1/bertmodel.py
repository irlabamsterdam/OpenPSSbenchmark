"""This file contains the implementation of a Dutch BERT model that
can be used for the automatic classification of documents based
on textual features. The file contains both the model itself as well as
a class for the pre processing of the data. The model can be used both
for training as well as for the classification of new documents based on a
model trained on other datasets. The scripts can be used to recreate the
runs from both EXPERIMENT_1 as well as EXPERIMENT_2"""


import os
import torch
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    Trainer, TrainingArguments

# Local imports
from utils import *
from metricutils import *
from dataloading import *
from utils import save_model_predictions


def main(arguments):
    # Set up the tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    # Load in both the train and test dataset
    training_params = load_data_for_experiment_1(arguments.train_dataset,
                                                 arguments.test_dataset,
                                                 'GUHA-TXT')

    train_dataframe = training_params['train_dataframe']
    test_dataframe = training_params['test_dataframe']
    pretrained_model_path = training_params['pretrained_model_path']

    train_text, train_labels = train_dataframe['text'].tolist(),\
                               train_dataframe['label'].astype(int).tolist()

    test_text, test_labels = test_dataframe['text'].tolist(),\
                             test_dataframe['label'].astype(int).tolist()

    # For the validation set, we are going to take out 20% of the training
    # corpus and use this as our validation set.
    train_text, val_text, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=.2)

    train_encodings = tokenizer(train_text, padding=True, truncation=True,
                                return_tensors='pt')
    val_encodings = tokenizer(val_text, padding=True, truncation=True,
                              return_tensors='pt')
    test_encodings = tokenizer(test_text, padding="max_length", truncation=True,
                               return_tensors='pt')

    train_data = WOBDatasetText(train_encodings, train_labels)
    test_data = WOBDatasetText(test_encodings, test_labels)
    val_data = WOBDatasetText(val_encodings, val_labels)

    if arguments.from_scratch:
        print('Training the BERT classification model from scratch')
        model = AutoModelForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased",
                                                                   num_labels=2).to(device)
    else:
        print('-- Evaluating with a previously trained model --')
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path,
                                                                   num_labels=2,
                                                                   output_hidden_states=True).to(device)

    # Set up the training arguments for the model
    training_args = TrainingArguments(
        output_dir='../logs',
        num_train_epochs=arguments.num_epochs,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        warmup_steps=100,
        weight_decay=0.005,
        logging_steps=10,
        save_strategy="no")

    if arguments.from_scratch:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data)
        trainer.train()
        trainer.save_model('../logs')
    else:
        trainer = Trainer(model)

    model_output = trainer.predict(test_data)

    model_predictions = np.argmax(model_output.predictions, axis=1).tolist()

    predictions_df = pd.DataFrame({'name': test_dataframe['name'].tolist(),
                                   'label': model_output.label_ids.tolist(),
                                   'prediction': model_predictions})

    # Get the predictions ad gold standard from the prediction dataframe
    predictions_dict = get_ground_truth_from_dataframe(predictions_df, 'label')
    gold_dict = get_ground_truth_from_dataframe(predictions_df, 'label')

    evaluation_report(gold_dict, predictions_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2'])
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['C1', 'C2', 'TOBACCO', 'C1C2_SAME', 'C1C2_DIFF'])
    args = parser.parse_args()

    main(args)

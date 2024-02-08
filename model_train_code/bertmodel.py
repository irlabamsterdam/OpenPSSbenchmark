"""This file contains the implementation of a Dutch BERT model that
can be used for the automatic classification of WOB documents based
on textual features. The file contains both the model itself as well as
a class for the pre processing of the data. The model can be used both
for training as well as for the classification of new documents based on a
model trained on other WOB datasets."""


import os
import torch
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    Trainer, TrainingArguments

# Local imports
from metricutils import *
import dataloading as dataloading


def main(args):

    # Write a small piece of code to check that we are indeed running on the GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("We are currently running on the %s" % device)

    # Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    # Load in both the train and test dataset
    train_df, test_df = dataloading.load_text_dataframe(args.train_path),\
                        dataloading.load_text_dataframe(args.test_path)

    train_text, train_labels = train_df['text'].tolist(),\
                               train_df['label'].astype(int).tolist()
    test_text, test_labels = test_df['text'].tolist(),\
                             test_df['label'].astype(int).tolist()

    # For the validation set, we are going to take out 20% of the training
    # corpus and use this as our validation set.
    train_text, val_text, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=.2)

    train_encodings = tokenizer(train_text, padding=True, truncation=True,
                                return_tensors='pt')
    val_encodings = tokenizer(val_text, padding=True, truncation=True,
                              return_tensors='pt')
    test_encodings = tokenizer(test_text, padding="max_length", truncation=True,
                               return_tensors='pt')

    train_data = dataloading.WOBDatasetText(train_encodings, train_labels)
    test_data = dataloading.WOBDatasetText(test_encodings, test_labels)
    val_data = dataloading.WOBDatasetText(val_encodings, val_labels)

    if args.train:
        print("-- Training a  vanilla BERT model --")
        model = AutoModelForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased", num_labels=2).to(device)
    else:
        print('-- Evaluating with a previously trained model --')
        model = AutoModelForSequenceClassification.from_pretrained(args.save_path, num_labels=2).to(
            device)

    # Set up the training arguments for the model
    training_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.005,
        logging_dir=args.log_dir,
        logging_steps=10,
        save_strategy="no"
    )

    if args.train:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data)
        trainer.train()
        trainer.save_model(args.save_path)
    else:
        trainer = Trainer(model)

    model_output = trainer.predict(test_data)
    model_predictions = np.argmax(model_output.predictions, axis=1).tolist()

    predictions_df = pd.DataFrame({'name': test_df['name'].tolist(),
                                   'label': model_output.label_ids.tolist(),
                                   'prediction': model_predictions})
    
    gold_dict = get_ground_truth_from_dataframe(predictions_df, 'label')
    predictions_dict = get_ground_truth_from_dataframe(predictions_df,
                                                       'prediction')
    print(gold_dict)
    print(predictions_dict)
    # We want to keep this score realistic, so for each first page in a stream
    # we set the class to be a one, as this is true by definition and thus
    # more accurately reflects the performance of the model.

    # TODO: This should probably also become a function in a separate file
    # so that we can keep this the same over multiple files.

    corrected_predictions_dict = {}
    for key, val in predictions_dict.items():
        corrected_predictions_dict[key] = val

    with open(os.path.join(args.save_path, 'gold.json'), 'w') as f:
        json.dump(gold_dict, f)
    with open(os.path.join(args.save_path, 'preds.json'), 'w') as f:
        json.dump(corrected_predictions_dict, f)

    results = calculate_mean_scores({key: np.array(val) for key, val in gold_dict.items()}, {key: np.array(val) for key, val in corrected_predictions_dict.items()})
    results.to_csv(os.path.join(args.save_path, args.result_location),
                   index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--result_location', type=str, default='results.csv')

    args = parser.parse_args()

    main(args)

"""This file contains the implementation of a Dutch BERT model that
can be used for the automatic classification of WOB documents based
on textual features. The file contains both the model itself as well as
a class for the pre processing of the data. The model can be used both
for training as well as for the classification of new documents based on a
model trained on other WOB datasets."""
import numpy as np
import tqdm
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local imports
from metricutils import *
import dataloading as dataloading


def main(args):

    # Write a small piece of code to check that we are indeed running on the GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("We are currently running on the %s" % device)

    # Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    print('-- Evaluating with a previously trained model --')
    model = AutoModelForSequenceClassification.from_pretrained(args.save_path, num_labels=2,
                                                               output_hidden_states=True).to(
        device)
    model.eval()
    # Load in both the train and test dataset
    test_df = dataloading.load_text_dataframe(args.test_path)

    cls_tokens = {}

    for doc_id, sub_df in tqdm.tqdm(test_df.groupby('name')):

        test_text, test_labels = sub_df['text'].tolist(),\
                                 sub_df['label'].astype(int).tolist()

        # For the validation set, we are going to take out 20% of the training
        # corpus and use this as our validation set.

        test_encodings = tokenizer(test_text, padding="max_length", truncation=True,
                                   return_tensors='pt')

        test_data = dataloading.WOBDatasetText(test_encodings, test_labels)
        stream_vectors = []

        for i, example in enumerate(test_data):
            with torch.no_grad():
                out = model(input_ids=example['input_ids'].unsqueeze(0).to(device), attention_mask=example['attention_mask'].unsqueeze(0).to(device),
                token_type_ids=example['token_type_ids'].unsqueeze(0).to(device))
                hidden_states = out['hidden_states']
                cls_token = hidden_states[-1][:, 0, :]
                stream_vectors.append(cls_token.detach().cpu())
        cls_tokens[doc_id] = torch.vstack(stream_vectors).cpu().numpy()

    print("prediction step finished, saving vectors")
    np.save(args.vector_path, cls_tokens)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--vector_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    main(args)

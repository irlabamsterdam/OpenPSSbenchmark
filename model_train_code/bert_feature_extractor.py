"""
Code to extract the features we got from BERT.
We extract the features by running the text of the pages through the trained
model and getting the representation of the CLS token as the representation
of the page that we will use.
"""

import os
import tqdm
import argparse
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from sentence_transformers import SentenceTransformer


def main(args):
    model = SentenceTransformer("GroNLP/bert-base-dutch-cased")

    vector_dict = {}

    dataframe = pd.read_csv(args.data_path)
    dataframe['text'] = dataframe['text'].fillna('UNK')

    # Loop through each sorted stream and get the vectors
    # Save using the numpy save format

    for stream_id, contents in dataframe.groupby('name'):
        stream_vectors = []
        # Make sure all these pages are sorted properly
        contents['page'] = contents['page'].astype(int)
        contents = contents.sort_values('page')
        for id_, row in tqdm.tqdm(contents.iterrows()):
            page_text = row['text']
            page_sentences = sent_tokenize(page_text)

            page_embedding = model.encode(page_sentences).mean(axis=0)
            if not page_embedding.shape:
                page_embedding = model.encode(['UNK']).mean(axis=0)
            stream_vectors.append(page_embedding)
        stream_vector_matrix = np.stack(stream_vectors)
        print(stream_vector_matrix.shape)

        vector_dict[stream_id] = stream_vector_matrix
        # save the vectors

    np.save(os.path.join(args.vector_save_path, 'vectors'), vector_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--vector_save_path', type=str, required=True)
    parser.add_argument('--method', type=str, default='average')

    arguments = parser.parse_args()
    main(arguments)

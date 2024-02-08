"""
This class uses the pretrained VGG16 model from torchvision to extract
visual features from the individual pages of the WOB documents.
"""

import os
import torch
import argparse
import tqdm
import numpy as np
from dataloading import *
from torchvision.models import vgg16, VGG16_Weights


def main(arguments):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1,
                       progress=True)

    model.classifier = model.classifier[:-1]

    model.eval()
    model.to(device)

    all_vectors = []

    dataset = WOBDatasetImage(arguments.data_path,
                              arguments.png_path)

    dataloader = DataLoader(dataset, batch_size=1,
                                 shuffle=False, num_workers=0)

    for sample, label in tqdm.tqdm(dataloader):
        with torch.no_grad():
            output = model(sample.to(device)).detach().cpu()
            all_vectors.append(output)

    stacked_vecs = torch.vstack(all_vectors)
    np.save(os.path.join(args.vector_save_path, 'pretrained.npy'), stacked_vecs.numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--png_path', required=True)
    parser.add_argument('--vector_save_path', required=False)

    args = parser.parse_args()

    main(args)

    

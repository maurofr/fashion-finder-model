"""
Script to get and the matrix of embeddings from a directory of images.
"""

from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from bad_paths import get_bad_paths
import pandas as pd

def main(args) -> None:
    # We load the pre-trained model for fashion embeddings and classification
    model = FashionCLIP('fashion-clip')

    # Directory containing all the jpg files to process (the masked images)
    directory = os.fsencode(args.inp_dir)
    img_paths = os.listdir(directory)

    # We save the paths to pass through the network
    bad_paths = get_bad_paths() # Some are broken
    paths = ['\\'.join((args.inp_dir,os.fsdecode(file))) for file in img_paths if os.fsdecode(file) not in bad_paths]


    # We get the embeddings and normalize them to get cosine similarity from dot product
    image_embeddings = model.encode_images(paths, batch_size=256)
    image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    
    # Load url dataframe
    all_links = pd.read_csv(args.data_path).values

    # Finally, we save the embeddings in a dataframe containing the links and the mean embedding
    embeds = {}
    for name, embedding in zip(img_paths, image_embeddings):
        name = int(name.split('_')[1])
        if name not in embeds:
            embeds[name] = []
        embeds[name].append(embedding)

    embeds = {name : list(np.mean(val, axis=0)) for name, val in embeds.items()}

    df = pd.DataFrame(embeds).transpose().rename(columns = {i : f'embedding_{i}' for i in range(512)}).reset_index()
    df['link_1'] = df['index'].map(lambda x: all_links[x][0])
    df['link_2'] = df['index'].map(lambda x: all_links[x][1])
    df['link_3'] = df['index'].map(lambda x: all_links[x][2])
    df = df.drop(columns='index').rename(columns={0:'embedding'})
    df.to_csv(args.out_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp_dir', type=str, help='Path to the images folder')
    parser.add_argument('-o', '--out_path', type=str, help='Path to the save embeddings')
    parser.add_argument('-d', '--data_path', type=str, help='Path to the urls csv')
    args = parser.parse_args()

    main(args)
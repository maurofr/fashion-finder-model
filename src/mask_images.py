"""
Script to mask all images in a folder and save them in a new one with the 
same image size and file names. It ran in about 7h for the whole dataset.
"""

from pipeline import ClothSegmentation
import os 
from tqdm import tqdm
from PIL import Image
import torch
import argparse

def main(args) -> None:
    # Load the segmentation model and set to evaluation
    seg_model = ClothSegmentation()
    seg_model.model.eval()

    # Directory of the images to process
    directory = os.fsencode(args.dir_str)
    # Path to the green screen for the background (black was a bad idea for some items)
    green_screen = Image.open(args.green_screen).convert('RGB')

    # Iterate over each image, mask it and save it
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        out_path = '\\'.join((args.out_dir,filename))
        # We check if the image has been already processed, 
        # in case we run the script more than once
        if not os.path.exists(out_path):
            try: # Handle broken images
                img = Image.open('\\'.join((args.dir_str,filename)))
            except:
                continue

            with torch.no_grad():
                masked_img = seg_model.mask_img(img, green_screen)

            masked_img.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_str', type=str, help='Path to the images folder')
    parser.add_argument('-o', '--out_dir', type=str, help='Path to the saves folder')
    parser.add_argument('-g', '--green_screen', type=str, help='Path to green screen')
    # parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to saved segmentation model')
    args = parser.parse_args()

    main(args)
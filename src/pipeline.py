"""
Script with some utilities and testing.
"""

from typing import Optional
from cloth_segmentation.process import load_seg_model, generate_mask, get_masked_img
from PIL import Image
import requests
import numpy as np

class ClothSegmentation():
    def __init__(self, palette: Optional[list] = None, checkpoint_path: Optional[str] = "cloth_segmentation/checkpoint/cloth_segm.pth", device: Optional[str] = "cuda"):
        """
        Load the pre-trained segmentation model, fine-tuned for clothing.

        :param palette: palette for the masks in the image (at most 4, the first being background)
        :param checkpoint_path: path where the trained .pth model is saved, if not found, it is downloaded (this functionality is not safe)
        :param device: torch device to compute the model and images
        """
        self.model = load_seg_model(checkpoint_path, device=device)
        self.palette = palette or [0]*3 + [255] * 9
        self.device = device

    def mask_img(self, img: Image, background: Image = None, palette: Optional[list] = None) -> Image:
        """
        Given a PIL Image of clothing, returns the masked version where all the background
        (anything that is not clothing) is black, and the clothes' colors and shapes are preserved.
        Usage of the default palette is recommended for this method.

        :param img: PIL Image to mask
        :param palette: palette to use, if None is passed the default of the class is used
        :returns: masked PIL image
        """
        palette = palette or self.palette
        img = img.convert('RGB')
        mask = generate_mask(img, net=self.model, palette=self.palette,device=self.device)
        return get_masked_img(img, mask, background)

def load_image(url: str) -> Image:
    """
    Load image from url address.

    :param url: the string containing the url
    :returns: PIL Image retrieved from the url
    """
    return Image.open(requests.get(url, stream=True).raw)

def generate_greenscreen(size) -> Image:
    img = Image.new('RGB', size)
    img_array = np.array(img)
    img_array[:,:,1].fill(255)
    img_array[:,:,2].fill(17)
    return Image.fromarray(img_array)


if __name__ == "__main__":
    # seg_model = ClothSegmentation()
    # url = "https://static.zara.net/photos///2023/I/0/3/p/9959/552/721/2/w/2048/9959552721_6_1_1.jpg?ts=1698929412065"
    # img = load_image(url)
    # masked_img = seg_model.mask_img(img=img)
    # masked_img.save(f"cloth_segmentation/output/{url[-8:]}_mask.jpg")
    generate_greenscreen((768, 768)).save("cloth_segmentation/data/green_screen.jpg")

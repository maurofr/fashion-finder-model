from typing import Optional
from cloth_segmentation.process import load_seg_model, generate_mask, get_masked_img
from PIL import Image

class ClothSegmentation():
    def __init__(self, palette: Optional[list] = None, checkpoint_path: Optional[str] = "cloth_segmentation/checkpoint/cloth_segm.pth", device: Optional[str] = "cuda"):
        """
        Load the pre-trained segmentation model, fine-tuned for clothing.

        :param palette: palette for the masks in the image (at most 4, the first being background)
        :param checkpoint_path: path where the trained .pth model is saved, if not found, it is downloaded (this functionality is not safe)
        :param device: torch device to compute the model and images
        """
        self.model = load_seg_model(checkpoint_path, device=device)
        self.palette = palette or [0] * 3 + [255] * 9
        self.device = device

    def mask_img(self, img: Image, palette: Optional[list] = None) -> Image:
        """
        Given a PIL Image of clothing, returns the masked version where all the background
        (anything that is not clothing) is black, and the clothes' colors and shapes are preserved.
        Usage of the default black-and-white palette is recommended for this method.

        :param img: PIL Image to mask
        :param palette: palette to use, if None is passed the default of the class is used
        :returns: masked PIL image
        """
        palette = palette or self.palette
        img = img.convert('RGB')
        mask = generate_mask(img, net=self.model, palette=self.palette,device=self.device)
        return get_masked_img(img, mask)

if __name__ == "__main__":
    seg_model = ClothSegmentation()

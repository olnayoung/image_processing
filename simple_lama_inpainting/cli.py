from models.model import SimpleLama
from PIL import Image
from pathlib import Path
import numpy as np
import cv2


def main():
    """Apply lama inpainting using given image and mask.

    Args:
        img_path (str): Path to input image (RGB)
        mask_path (str): Path to input mask (Binary 1-CH Image.
                        Pixels with value 255 will be inpainted)
        out_path (str, optional): Optional output imaga path.
                        If not provided it will be saved to the same
                            path as input image.
                        Defaults to None.
    """
    image_path = "/data/docker/olnayoung/image_processing/imgs/dogsNcats.jpg"
    mask_path = "/data/docker/olnayoung/image_processing/result/5.png"
    out_path = "/data/docker/olnayoung/image_processing/dogsNcats_result.jpg"

    image_path = Path(image_path)
    mask_path = Path(mask_path)

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask_array = np.array(mask)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=2)
    mask = Image.fromarray(dilated_mask)

    assert img.mode == "RGB" and mask.mode == "L"

    lama = SimpleLama()
    result = lama(img, mask)

    Path.mkdir(Path(out_path).parent, exist_ok=True, parents=True)
    result.save(out_path)
    print(f"Inpainted image is saved to {out_path}")



if __name__ == "__main__":
    main()

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token("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")

import glob
import os

import cv2
import numpy as np
from PIL import Image

from est_models.models.segformer import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


@click.command()
@click.option(
    "--img_dir",
    "img_dir",
    help="imgs directory to segment",
    required=True,
    metavar="DIR",
)
@click.option(
    "--mask_outdir",
    "mask_outdir",
    help="mask directory to save",
    required=True,
    metavar="DIR",
)
def main(img_dir, mask_outdir):
    processor = (
        SegformerImageProcessor()
    )  # processor to convert image to tensor, tensor to image
    model = SegformerForSemanticSegmentation.from_pretrained(
        "ai-human-lab/segformer-b5-finetuned-occludedface"
    )
    model.to("cuda")

    for path in sorted(glob.glob(f"{img_dir}/*")):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        img = Image.fromarray((img).astype(np.uint8))
        input_dict = processor(img, return_tensors="pt").to("cuda")
        output_dict = model(**input_dict)
        binary_mask_list = processor.post_process_binary_segmentation(
            output_dict, target_sizes=[img.size[::-1]]
        )
        cv2.imwrite(f"{mask_outdir}/{fname}", binary_mask_list[0])


if __name__ == "__main__":
    main()


# python segformer_run.py --img_dir [DIR] --mask_outdir [DIR]

from PIL import Image
import cv2
import numpy as np

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')

image = Image.open("./tests/images/OG2.jpg")

def test_Segformer():
    try:
        from est_models.models.segformer import SegformerForSemanticSegmentation, SegformerImageProcessor
        
        processor = SegformerImageProcessor() #processor to convert image to tensor, tensor to image
        model = SegformerForSemanticSegmentation.from_pretrained("ai-human-lab/segformer-b0-finetuned-occludedface")
        model.to("cuda")

        input_dict = processor(image, return_tensors="pt").to("cuda")
        output_dict = model(**input_dict)
        binary_mask_list = processor.post_process_binary_segmentation(output_dict, target_sizes=[image.size[::-1]])

    except Exception as e:
        raise e

def test_PPMattingV2():
    try:
        from est_models.models.ppmattingv2 import ORTPPMattingV2ForHumanMatting, PPMattingV2ImageProcessor

        processor = PPMattingV2ImageProcessor()
        matt_model = ORTPPMattingV2ForHumanMatting.from_pretrained("ai-human-lab/PPMattingV2")
        preprocess_dict = processor([image])
        output = matt_model(np.array(preprocess_dict["pixel_values"]))
        output = processor.post_process_human_matting(output)
    except Exception as e:
        raise e

def test_BiSeNetV2_BinarySegmentation():
    try:
        from est_models.models.bisenetv2 import BiSeNetV2ForSemanticSegmentation, BiseNetV2ImageProcessor

        processor = BiseNetV2ImageProcessor() #processor to convert image to tensor, tensor to image
        model = BiSeNetV2ForSemanticSegmentation.from_pretrained( "ai-human-lab/bisenetv2-hair-segmentation")
        model.to("cuda")

        input_dict = processor(image, return_tensors="pt").to("cuda")
        output_dict = model(**input_dict)
        binary_mask_list = processor.post_process_binary_segmentation(output_dict, target_sizes=[image.size[::-1]])

    except Exception as e:
        raise e

def test_BiSeNetV2_SemanticSegmentation():
    try:
        from est_models.models.bisenetv2 import BiSeNetV2ForSemanticSegmentation, BiseNetV2ImageProcessor

        processor = BiseNetV2ImageProcessor() #processor to convert image to tensor, tensor to image
        model = BiSeNetV2ForSemanticSegmentation.from_pretrained("ai-human-lab/bisenetv2-face-parsing")
        model.to("cuda")

        input_dict = processor(image, return_tensors="pt").to("cuda")
        output_dict = model(**input_dict)
        mask_list = processor.post_process_semantic_segmentation(output_dict, target_sizes=[image.size[::-1]])

        #nose mask
        binary_mask = np.where(mask_list[0].cpu()==model.config.label2id["nose"], 255, 0)


    except Exception as e:
        raise e
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import mediapipe as mp
from utils import paste_image, get_mask_else, LEFT_EYE_TIGHT, RIGHT_EYE_TIGHT, MOUTH_TIGHT

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything2.sam2.build_sam import build_sam2
from segment_anything2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything2.sam2.sam2_image_predictor import SAM2ImagePredictor
from simple_lama_inpainting.models.model import SimpleLama
from est_models.models.segformer import SegformerForSemanticSegmentation, SegformerImageProcessor
from DIS.ISNet.models import *

import os

class ImageParams():
    def __init__(self):
        pass

    # 대비
    def contrast(self, img, value, intensity):
        # result = cv2.addWeighted(img, (1.0 + value * intensity), np.zeros(img.shape, img.dtype), 0, 0)

        img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img)
        result = enhancer.enhance(1.0 + value * intensity)  # 대비 -0% ~ -25%
        return np.array(result)

    # 채도
    def saturation(self, img, value, intensity):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + value * intensity), 0, 255).astype(np.uint8)  # 채도 40% * intensity 증가
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # img = Image.fromarray(img)
        # enhancer = ImageEnhance.Color(img)
        # result = enhancer.enhance(1.0 + value * intensity) 
        return result

    # 색조
    def color(self, img, value, intensity):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] + value * intensity
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return result
    
    # 그림자
    def shadow(self, img, value, intensity):
        result = np.where(img < 128, img + (value * intensity), img).astype(np.uint8)
        return result

    # 하이라이트
    def hilight(self, img, value, intensity):
        result = cv2.addWeighted(img, 1.0, np.zeros(img.shape, img.dtype), 0, value * intensity)
        return result

    # 블랙포인트
    def black_point(self, img, value, intensity):
        result = np.where(img < value * intensity, 0, img - value * intensity).astype(np.uint8)
        return result

    # 선명도
    def sharpness(self, img, value, intensity):
        pil_image = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(pil_image)
        result = enhancer.enhance(1 + value * intensity)  # 선명도를 조절
        return np.array(result)
    
    # 노출, 휘도, 밝기
    def exposure(self, img, value, intensity):
        result = np.clip(img * (1.0 + value * intensity), 0, 255).astype(np.uint8)
        return result

im = ImageParams()

def apply_backlight_filters(image_array, intensity=1.0):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    img = im.exposure(img, 0.3, intensity)      # 밝기
    img = im.contrast(img, -0.2, intensity)     # 대비
    img = im.saturation(img, 0.8, intensity)    # 채도
    img = im.hilight(img, 0.3, intensity)       # 하이라이트
    img = im.black_point(img, 0.5, intensity)
    img = im.sharpness(img, 0.3, intensity)    # 선명도

    final_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return final_img

def apply_sunset_filters(image_array, intensity=1.0):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    img = im.exposure(img, 0.1, intensity) 
    img = im.contrast(img, -0.3, intensity)     # 대비
    img = im.saturation(img, 0.5, intensity)    # 채도
    # img = im.shadow(img, -10, intensity)        # 그림자
    img = im.sharpness(img, 0.1, intensity)

    transform_matrix = np.array([
        [1.0 + 0.05 * intensity, 0.0, 0.0],  # 빨간색 채널 조정
        [0.0, 0.95, 0.0],  # 녹색 채널 조정
        [0.0, 0.0, 1.0]
    ])
    img = cv2.transform(img, transform_matrix.astype(np.float32))

    final_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return final_img

def apply_cherryblossom_filters(image_array, intensity=1.0):

    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    # 노출 -5% ~ 0%, 휘도 +0% ~ 20%, 밝기 +0% ~ 10%
    factor = (0.95 + 0.05 * intensity) * (1.0 + 0.2 * intensity) * (1.0 + 0.1 * intensity)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    img = im.hilight(img, -10, intensity)       # 하이라이트
    img = im.shadow(img, 20, intensity)         # 그림자
    img = im.contrast(img, -0.25, intensity)    # 대비
    img = im.black_point(img, 10, intensity)    # 블랙포인트
    img = im.saturation(img, 0.3, intensity)    # 채도

    # 따뜻함 조정 (색조 변환을 위한 3x3 변환 행렬 사용)
    img = np.array(img)
    transform_matrix = np.array([
        [1.0 + 0.05 * intensity, 0.0, 0.0],  # 빨간색 채널 조정
        [0.0, 0.95, 0.0],  # 녹색 채널 조정
        [0.0, 0.0, 1.0]
    ])
    img = cv2.transform(img, transform_matrix.astype(np.float32))

    img = im.color(img, 10, intensity)          # 색조
    img = im.sharpness(img, 0.1, intensity)     # 선명도

    final_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return final_img

class ChangeMood():
    def __init__(self):
        self.effect_matrix = {
            "sunset": [[0.5, 0, 0],
                       [0, 0.7, 0],
                       [0, 0, 1.2]],
            "warm": [[0.7, 0, 0],
                     [0, 1.3, 0],
                     [0, 0, 1.5]],
            "cool": [[1.5, 0, 0],
                     [0, 1.1, 0],
                     [0, 0, 0.7]],
            "vintage": [[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]],
            "grayscale": [[0.299, 0.587, 0.114],
                         [0.299, 0.587, 0.114],
                         [0.299, 0.587, 0.114]]
        }

        
    def __call__(self, image, effect_type, intensity):
        if effect_type not in self.effect_matrix.keys():
            raise ValueError("Unknown effect type.")

        effect_matrix = np.array(self.effect_matrix[effect_type])

        effect_matrix = np.eye(3) * (1 - intensity) + effect_matrix * intensity
        image = cv2.transform(image, effect_matrix)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image


class FaceBeauty():
    def __init__(self):
        self.seg_processor = SegformerImageProcessor()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(f'./model/segformer_b5').to("cuda")

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3, # 0.5 -> 0.3
            min_tracking_confidence=0.5,
            )

        kernel_size = (7, 7)
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    def get_face_mask(self, image):
        input_dict = self.seg_processor(image, return_tensors="pt").to("cuda")
        output_dict = self.segformer(**input_dict)
        binary_mask_list = self.seg_processor.post_process_binary_segmentation(output_dict, target_sizes=[image.size[::-1]])
        mask = cv2.cvtColor(binary_mask_list[0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return mask

    def get_mask_wo_eyes_lips(self, image):
        results = self.face_mesh.process(image)
        face_landmarks = results.multi_face_landmarks[0]
        mask_img = np.zeros(image.shape, dtype=np.uint8)

        mask_Leye = get_mask_else(face_landmarks, image, mask_img, [LEFT_EYE_TIGHT], alpha_blend=False, avg=False)
        mask_Reye = get_mask_else(face_landmarks, image, mask_img, [RIGHT_EYE_TIGHT], alpha_blend=False, avg=False)
        mask_mouth = get_mask_else(face_landmarks, image, mask_img, [MOUTH_TIGHT], alpha_blend=False, avg=False)
        
        face_mask = self.get_face_mask(Image.fromarray(image))
        face_mask = cv2.bitwise_xor(face_mask, mask_Leye)
        face_mask = cv2.bitwise_xor(face_mask, mask_Reye)
        face_mask = cv2.bitwise_xor(face_mask, mask_mouth)
        
        dilated_mask = cv2.dilate(face_mask, self.dilate_kernel, iterations=1)
        blurred_mask = cv2.GaussianBlur(dilated_mask, (19, 19), 0)
        final_mask = cv2.max(blurred_mask, face_mask)

        return final_mask
    
    def __call__(self, img, gamma, strength=50, hsv_start_x=0, hsv_start_y=10, hsv_start_z=150, hsv_end_x=200, hsv_end_y=255, hsv_end_z=255):
        # smooth face
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_img, np.array((hsv_start_x, hsv_start_y, hsv_start_z)), np.array((hsv_end_x, hsv_end_y, hsv_end_z)))
        
        full_mask = cv2.merge((hsv_mask, hsv_mask, hsv_mask))
        blurred_img = cv2.bilateralFilter(img, 5, strength, strength)
        blurred_img = cv2.medianBlur(blurred_img, 3)

        masked_img = cv2.bitwise_and(blurred_img, full_mask)
        inverted_mask = cv2.bitwise_not(full_mask)
        masked_img2 = cv2.bitwise_and(img, inverted_mask)
        smoothed_img = cv2.add(masked_img2, masked_img)

        # do gamma correction
        gc_img = (((smoothed_img / 255) ** (1 / gamma)) * 255).astype(np.uint8)
        
        # get face mask without eyes and lip
        final_mask = self.get_mask_wo_eyes_lips(smoothed_img)

        result = paste_image(gc_img, img, final_mask)
        
        return result


class SegmentImage2():
    def __init__(self, version="SAM2"):
        if version == "SAM2":
            sam2_checkpoint = "./model/SAM2/sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(sam2)

        elif version == "SAM1":
            model_type = "vit_b"
            checkpoint_path = "./model/SAM/sam_vit_b_01ec64.pth"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device="cuda")
            self.generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

    
    def __call__(self, image):
        h, w, _ = image.shape
        centerH, centerW = h//2, w//2
        # input_point = np.array([[centerW, centerH]])
        input_point = np.array([[0.5, 0.5]])
        input_label = np.array([1])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            normalize_coords=False
        )
        threshold = 0.5
        print(scores)
        # masks = [masks[i] for i in range(len(scores)) if scores[i] > threshold]

        return masks


class SegmentImage():
    def __init__(self, version="SAM2"):
        if version == "SAM2":
            sam2_checkpoint = "./model/SAM2/sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
            self.generator = SAM2AutomaticMaskGenerator(sam2,
                                                        pred_iou_thresh=0.7,
                                                        stability_score_offset=0.8,
                                                        # points_per_side=None,
                                                        # point_grids=[np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                                                        # point_grids=[np.array([0])]
                                                        )

        elif version == "SAM1":
            model_type = "vit_b"
            checkpoint_path = "./model/SAM/sam_vit_b_01ec64.pth"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device="cuda")
            self.generator = SamAutomaticMaskGenerator(sam, 
                                                       pred_iou_thresh=0.7,
                                                       stability_score_offset=0.8,
                                                    #    points_per_side=None,
                                                    #    point_grids=[np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                                                    #    point_grids=[np.array([0])],
                                                       output_mode="binary_mask")

    
    def __call__(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.generator.generate(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(masks)

        return [mask_data["segmentation"] for mask_data in masks]


class InpaintEmpty():
    def __init__(self):
        self.lama = SimpleLama()
    
    def __call__(self, image, mask, dilate_iter=2):
        img = Image.fromarray(image)
        kernel = np.ones((5, 5), np.uint8)
        mask_array = mask.astype(np.uint8) * 255
        dilated_mask = cv2.dilate(mask_array, kernel, iterations=dilate_iter)
        mask = Image.fromarray(dilated_mask).convert("L")

        assert img.mode == "RGB" and mask.mode == "L"

        result = self.lama(img, mask)
        result = np.array(result)
        originH, originW, _ = image.shape
        result = result[:originH, :originW, :]

        return result

    def resizeElement(self, image, mask, dilate_iter, factor):
        mask = mask.astype(np.uint8) * 255
        element = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(mask)

        element_crop = element[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        new_w, new_h = int(factor*w), int(factor*h)
        element_resized = cv2.resize(element_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        new_x = max(x + int(w*(1-factor)/2), 0)
        new_y = max(y + int(h*(1-factor)/2), 0)

        filled_image = self(image, mask, dilate_iter)

        output_image = filled_image.copy()
        output_image[new_y:new_y+new_h, new_x:new_x+new_w] = element_resized
        output_mask = np.zeros((mask.shape))
        output_mask[new_y:new_y+new_h, new_x:new_x+new_w] = mask_resized
        output_mask = np.stack((output_mask, output_mask, output_mask), axis=-1)

        result = paste_image(output_image, filled_image, output_mask)

        return result


class removeBG():
    def __init__(self):
        model_path = "./DIS/saved_models/isnet-general-use.pth"  # the model path
        self.input_size=[1024,1024]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = ISNetDIS()
        self.net.load_state_dict(torch.load(model_path, weights_only=True))
        self.net = self.net.to(self.device)
        self.net.eval()
    
    def __call__(self, image, threshold):
        
        im_shp = image.shape[0:2]
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), self.input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0])
        image = image.to(self.device)

        mask = self.net(image)
        mask = torch.squeeze(F.interpolate(mask[0][0],im_shp,mode='bilinear'),0)
        ma = torch.max(mask)
        mi = torch.min(mask)
        mask = (mask-mi)/(ma-mi)

        mask = (mask*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        mask[mask > threshold] = 255

        return mask/255
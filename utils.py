import cv2
import skimage
import numpy as np

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from dataclasses import dataclass

class CustomIndex:
    def __init__(self, index: list = [], info: str = None):
        self.index = index
        self.info = info

    def __repr__(self):
        return ",".join(map(str, self.index))

LEFT_EYE_TIGHT = CustomIndex(index=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
RIGHT_EYE_TIGHT = CustomIndex(index=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,])
MOUTH_TIGHT = CustomIndex(index=[0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37])

def center_length_of_array(cords):
    xs, ys = list(zip(*cords))

    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    lx = (max(xs) - min(xs)) / 2
    ly = (max(ys) - min(ys)) / 2

    return cx, cy, lx, ly

def get_points(face_landmarks, crop_image, masktype, avg):
    cords_li = []
    for indexes in masktype:
        cords = []
        for lip_id in indexes if avg else indexes.index:
            lid = lip_id if avg else face_landmarks.landmark[lip_id]
            
            lid.x = np.clip(lid.x, 0.01, 0.9999)
            lid.y = np.clip(lid.y, 0.01, 0.9999)

            cords.append(
                _normalized_to_pixel_coordinates(
                    lid.x,
                    lid.y,
                    crop_image.shape[1],
                    crop_image.shape[0],
                )
            )

        cords_li.append(cords)

    return cords_li


def get_mask_else(face_landmarks, crop_image, mask_img, masktype, alpha_blend, avg):
    cords_li = get_points(face_landmarks, crop_image, masktype, avg)

    for cords in cords_li:
        cx, cy, lx, ly = center_length_of_array(cords)
        Y, X = skimage.draw.polygon(list(zip(*cords))[1], list(zip(*cords))[0])

        if alpha_blend:
            for i in range(len(X)):
                mask_img[Y[i], X[i]] = min(
                    (max((ly - abs(cy - Y[i])), 0) / ly) * 255,
                    (max((lx - abs(cx - X[i])), 0) / lx) * 255,
                )

        else:
            mask_img[Y, X] = 255

    return mask_img


def paste_image(foreground, background, mask):
    if len(mask.shape) == 2:
        mask = np.stack((mask, mask, mask), axis=-1)
    normalized_mask = mask.astype(np.float32) / 255.0
    inverse_normalized_mask = 1.0 - normalized_mask

    foreground = foreground.astype(np.float32)
    background = background.astype(np.float32)

    foreground_weighted = cv2.multiply(foreground, normalized_mask)
    background_weighted = cv2.multiply(background, inverse_normalized_mask)

    result = cv2.add(foreground_weighted, background_weighted)
    result /= 255

    return result

def cut_object(image, mask):
    mask = mask.astype(np.uint8)
    object_A = cv2.bitwise_and(image, image, mask=mask)
    
    b, g, r = cv2.split(object_A)
    alpha = np.where(mask == 1, 255, 0).astype(np.uint8)
    rgba = cv2.merge([b, g, r, alpha])
    return rgba
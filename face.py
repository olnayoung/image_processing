import numpy as np
import cv2

from process import apply_backlight_filters


image = cv2.imread("./imgs/2people.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = apply_backlight_filters(image, 1)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_image.png", image)


from scipy.spatial import ConvexHull

def find_circle_center_and_radius(p1, p2, p3):
    # p1, p2, p3는 각각 (x, y) 좌표
    A = np.array([
        [p1[0] - p2[0], p1[1] - p2[1]],
        [p1[0] - p3[0], p1[1] - p3[1]]
    ])
    B = np.array([
        (p1[0]**2 - p2[0]**2 + p1[1]**2 - p2[1]**2) / 2,
        (p1[0]**2 - p3[0]**2 + p1[1]**2 - p3[1]**2) / 2
    ])

    # A * [cx, cy] = B를 푸는 과정
    center = np.linalg.solve(A, B)
    cx, cy = center[0], center[1]

    # 반지름 계산
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    return (int(cx), int(cy)), int(radius)

def draw_circle_from_normalized_points(image, points):
    height, width, _ = image.shape
    
    scaled_points = [(int(p[0] * width), int(p[1] * height)) for p in points]

    center_point = np.mean(scaled_points, axis=0)
    distances = [np.linalg.norm(np.array(p) - center_point) for p in scaled_points]

    farthest_indices = np.argsort(distances)[-3:]
    farthest_points = [scaled_points[i] for i in farthest_indices]

    center, radius = find_circle_center_and_radius(*farthest_points)

    axes = (radius, int(radius * 1.2))
    # cv2.ellipse(image, center, axes, 0, 0, 360, (255, 0, 0), 2)

    # for p in scaled_points:
    #     cv2.circle(image, p, 5, (0, 0, 255), -1)

    mask = np.zeros_like(image, dtype=np.uint8)
    # cv2.circle(mask, center, radius, (255, 255, 255), -1)  # 원형 마스크 생성
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    blurred_image = cv2.GaussianBlur(image, (13, 13), 11)  # 전체 블러 적용

    # 원 내부는 블러된 이미지, 외부는 원본 이미지로 합성
    result = np.where(mask == 255, blurred_image, image)

    return result


image = cv2.imread("./imgs/2people.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='./model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)
with FaceDetector.create_from_options(options) as detector:
    face_detector_result = detector.detect(mp_image)
    print(len(face_detector_result.detections))

    for detection in face_detector_result.detections:
        keypoints = detection.keypoints
    
        points = [(k.x, k.y) for k in keypoints]

        image = draw_circle_from_normalized_points(image, points)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_image.png", image)
<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
<br>

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), developed by [Ultralytics](https://ultralytics.com),
is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces
new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image
classification tasks.

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>
</div>

## 🔥Update


- ✅ **YOLOv8-n (person) trained on WIDERPedestrian [03.03]** 
- ✅ **YOLOv8-m (face) trained on WIDERFace [23.10]** 
- ✅ **YOLOv8-l (face) trained on WIDERFace [23.10]** 

## Installation

``` shell
# clone repo
git clone https://github.com/akanametov/yolov8-face

# pip install required packages
pip install ultralytics

# go to code folder
cd yolov8-face
```

## Trained models

[`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt)
[`yolov8m-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt)
[`yolov8l-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt)

[`yolov8n-person.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-person.pt)

[`yolov8n-football.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-football.pt)
[`yolov8m-football.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-football.pt)

[`yolov8n-parking.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-parking.pt)
[`yolov8m-parking.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-parking.pt)

[`yolov8n-drone.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-drone.pt)
[`yolov8m-drone.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-drone.pt)

</details>

# YOLOv8-face

## Inference

On image:

```shell
yolo task=detect mode=predict model=yolov8n-face.pt conf=0.25 imgsz=1280 line_thickness=1 max_det=1000 source=examples/face.jpg
```

<div align="center">
    <a href="./">
        <img src="./results/face/face.jpg" width="90%"/>
    </a>
</div>

## Results

PR curve:
<div align="center">
    <a href="./">
        <img src="./results/face/P_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/face/PR_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/face/R_curve.png" width="30%"/>
    </a>
</div>

Losses and mAP:
<div align="center">
    <a href="./">
        <img src="./results/face/results.png" width="80%"/>
    </a>
</div>

Confusion matrix:
<div align="center">
    <a href="./">
        <img src="./results/face/confusion_matrix.png" width="70%"/>
    </a>
</div>

## Training

Data preparation

* Download [dataset](http://shuoyang1213.me/WIDERFACE/):

* Download pretrained [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) model.

Single GPU training

``` shell
# train model
yolo task=detect \
mode=train \
model=yolov8n.pt \
data=datasets/data.yaml \
epochs=100 \
imgsz=640
```

# YOLOv8-person

## Inference

On image:

```shell
yolo task=detect mode=predict model=yolov8n-face.pt conf=0.25 imgsz=1280 line_thickness=1 max_det=1000 source=examples/person.jpg
```

<div align="center">
    <a href="./">
        <img src="./results/person/person.jpg" width="90%"/>
    </a>
</div>

## Results

PR curve:
<div align="center">
    <a href="./">
        <img src="./results/person/P_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/person/PR_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/person/R_curve.png" width="30%"/>
    </a>
</div>

Losses and mAP:
<div align="center">
    <a href="./">
        <img src="./results/person/results.png" width="80%"/>
    </a>
</div>

## Training

Data preparation

* Download [dataset](https://competitions.codalab.org/competitions/19118):

* Download pretrained [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) model.

Single GPU training

``` shell
# train model
yolo task=detect \
mode=train \
model=yolov8n.pt \
data=datasets/data.yaml \
epochs=100 \
imgsz=640
```

# YOLOv8-football

## Inference

On image:

```shell
yolo task=detect mode=predict model=yolov8m-football.pt conf=0.25 imgsz=1280 line_thickness=1 source=examples/football.jpg
```

<div align="center">
    <a href="./">
        <img src="./results/football/football.jpg" width="90%"/>
    </a>
</div>

## Results

PR curve:
<div align="center">
    <a href="./">
        <img src="./results/football/P_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/football/PR_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/football/R_curve.png" width="30%"/>
    </a>
</div>

Losses and mAP:
<div align="center">
    <a href="./">
        <img src="./results/football/results.png" width="80%"/>
    </a>
</div>

Confusion matrix:
<div align="center">
    <a href="./">
        <img src="./results/football/confusion_matrix.png" width="70%"/>
    </a>
</div>

## Training

Data preparation

* Download [dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/2#):

* Download pretrained [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) model.

Single GPU training

``` shell
# train model
yolo task=detect \
mode=train \
model=yolov8m.pt \
data=datasets/data.yaml \
epochs=120 \
imgsz=960
```
# YOLOv8-parking

## Inference

On image:

```shell
yolo task=detect mode=predict model=yolov8m-parking.pt conf=0.25 imgsz=1280 line_thickness=1 source=examples/parking.jpg
```

<div align="center">
    <a href="./">
        <img src="./results/parking/parking.jpg" width="90%"/>
    </a>
</div>

## Results

PR curve:
<div align="center">
    <a href="./">
        <img src="./results/parking/P_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/parking/PR_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/parking/R_curve.png" width="30%"/>
    </a>
</div>

Losses and mAP:
<div align="center">
    <a href="./">
        <img src="./results/parking/results.png" width="80%"/>
    </a>
</div>

Confusion matrix:
<div align="center">
    <a href="./">
        <img src="./results/parking/confusion_matrix.png" width="70%"/>
    </a>
</div>

## Training

Data preparation

* Download [dataset](https://universe.roboflow.com/brad-dwyer/pklot-1tros/dataset/4):

* Download pretrained [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) model.

Single GPU training

``` shell
# train model
yolo task=detect \
mode=train \
model=yolov8m.pt \
data=datasets/data.yaml \
epochs=10 \
batch=32 \
imgsz=640
```

# YOLOv8-drone

## Inference

On image:

```shell
yolo task=detect mode=predict model=yolov8m-drone.pt conf=0.25 imgsz=1280 line_thickness=1 source=examples/drone.jpg
```

<div align="center">
    <a href="./">
        <img src="./results/drone/drone.jpg" width="90%"/>
    </a>
</div>

## Results

PR curve:
<div align="center">
    <a href="./">
        <img src="./results/drone/P_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/drone/PR_curve.png" width="30%"/>
    </a>
    <a href="./">
        <img src="./results/drone/R_curve.png" width="30%"/>
    </a>
</div>

Losses and mAP:
<div align="center">
    <a href="./">
        <img src="./results/drone/results.png" width="80%"/>
    </a>
</div>

Confusion matrix:
<div align="center">
    <a href="./">
        <img src="./results/drone/confusion_matrix.png" width="70%"/>
    </a>
</div>

## Training

Data preparation

* Download [dataset](https://universe.roboflow.com/projects-s5hzp/dronesegment/dataset/1):

* Download pretrained [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) model.

Single GPU training

``` shell
# train model
yolo task=detect \
mode=train \
model=yolov8m.pt \
data=datasets/data.yaml \
epochs=100 \
imgsz=640
```

## Transfer learning

[`yolov8n.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)

[`yolov8m.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

## <div align="center">License</div>

YOLOv8 is available under two different licenses:

- **GPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source
  requirements of GPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and
  applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).

## <div align="center">Contact</div>

For YOLOv8 bugs and feature requests please visit [GitHub Issues](https://github.com/ultralytics/ultralytics/issues).
For professional support please [Contact Us](https://ultralytics.com/contact).

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-producthunt.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-facebook.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="" /></a>
</div>

# How to Use models


## Segformer (Binary Segmentation)

```python
from est_models.models.segformer import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import cv2
import numpy as np

processor = SegformerImageProcessor() #processor to convert image to tensor, tensor to image
model = SegformerForSemanticSegmentation.from_pretrained("ai-human-lab/segformer-b0-finetuned-occludedface")
model.to("cuda")

# local에서 로드하기
#model = SegformerForSemanticSegmentation.from_pretrained("./segformer-b0-finetuned-occludedface")

img = Image.open("../HairMapper/hairs/C컬펌_단발1.jpg")
input_dict = processor(img, return_tensors="pt").to("cuda")
output_dict = model(**input_dict)
binary_mask_list = processor.post_process_binary_segmentation(output_dict, target_sizes=[img.size[::-1]])
Image.fromarray(binary_mask_list[0])
```

## PPMattingV2(Human Matting)
transformers>=4.34.0
```python
from est_models.models.ppmattingv2 import ORTPPMattingV2ForHumanMatting, PPMattingV2ImageProcessor
from PIL import Image
import numpy as np

img = Image.open("../../BiSeNet/est_001/001.jpg")

processor = PPMattingV2ImageProcessor()
matt_model = ORTPPMattingV2ForHumanMatting.from_pretrained("ai-human-lab/PPMattingV2")
preprocess_dict = processor([img])
output = matt_model(np.array(preprocess_dict["pixel_values"]))
output = processor.post_process_human_matting(output, target_sizes=[img.size[::-1]])
Image.fromarray((output[0]*255).astype(np.uint8))
```


## BiSeNetV2 (Binary Segmentation)

```python
from est_models.models.bisenetv2 import BiSeNetV2ForSemanticSegmentation, BiseNetV2ImageProcessor
from PIL import Image
import cv2
import numpy as np

processor = BiseNetV2ImageProcessor() #processor to convert image to tensor, tensor to image
model = BiSeNetV2ForBinarySegmentation.from_pretrained( "ai-human-lab/bisenetv2-hair-segmentation")
model.to("cuda")

# local에서 로드하기
#model = BiSeNetV2ForBinarySegmentation.from_pretrained("./bisenetv2-hair-segmentation").to("cuda")

img = Image.open("../HairMapper/hairs/C컬펌_단발1.jpg")
input_dict = processor(img, return_tensors="pt").to("cuda")
output_dict = model(**input_dict)
binary_mask_list = processor.post_process_binary_segmentation(output_dict, target_sizes=[img.size[::-1]])
Image.fromarray(binary_mask_list[0])
```

## BiSeNetV2 (Semantic Segmentation)

```python
from est_models.models.bisenetv2 import BiSeNetV2ForSemanticSegmentation, BiseNetV2ImageProcessor
from PIL import Image
import cv2
import numpy as np

processor = BiseNetV2ImageProcessor() #processor to convert image to tensor, tensor to image
model = BiSeNetV2ForBinarySegmentation.from_pretrained("ai-human-lab/bisenetv2-face-parsing")
model.to("cuda")

# local에서 로드하기
#model = BiSeNetV2ForBinarySegmentation.from_pretrained("./bisenetv2-face-parsing").to("cuda")

img = Image.open("../HairMapper/hairs/C컬펌_단발1.jpg")
input_dict = processor(img, return_tensors="pt").to("cuda")
output_dict = model(**input_dict)
mask_list = processor.post_process_semantic_segmentation(output_dict, target_sizes=[img.size[::-1]])

#nose mask
binary_mask = np.where(mask_list[0].cpu()==model.config.label2id["nose"], 255, 0)
Image.fromarray(binary_mask.astype('uint8'))
```

import gradio as gr
import numpy as np

from process import *
from utils import cut_object

segmentImage2 = SegmentImage2()
inpaintEmpty = InpaintEmpty()
removebg = removeBG()

def get_xy(image, x_ratio, y_ratio):
    h, w, _ = image.shape
    newH, newW = int(h*y_ratio), int(w*x_ratio)
    return newH, newW


def image2predictor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmentImage2.predictor.set_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def get_object_mask(image, x_ratio, y_ratio):
    newH, newW = get_xy(image, x_ratio, y_ratio)
    input_point = np.array([[newW, newH]])
    input_label = np.array([1])
    
    masks, scores, logits = segmentImage2.predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    if len(masks) > 0:
        threshold = 0.9
        print(scores)
        masks = [masks[i] for i in range(len(scores)) if scores[i] > threshold]

    return masks
    
def remove_object(image, x_ratio, y_ratio, n_dilate=2):
    newH, newW = get_xy(image, x_ratio, y_ratio)
    masks = get_object_mask(image, x_ratio, y_ratio)

    if len(masks) == 0:
        cv2.circle(image, (newW, newH), radius=10, color=(0, 0, 255), thickness=-1)
        return image, None
    combined_mask = np.maximum.reduce(masks)
    object = cut_object(image, combined_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=n_dilate)

    image = inpaintEmpty(image, combined_mask)

    # cv2.circle(image, (newW, newH), radius=10, color=(255, 0, 0), thickness=-1)
    return image, object


def move_object(image, x_ratio, y_ratio, moveX, moveY):
    global masks
    y, x = get_xy(image, x_ratio, y_ratio)
    y_offset, x_offset = get_xy(image, moveX, moveY)

    masks = get_object_mask(image, x_ratio, y_ratio)
    if len(masks) == 0:
        cv2.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
        return image
    combined_mask = np.maximum.reduce(masks)

    image_B, object_A = remove_object(image, x_ratio, y_ratio)
    h, w, _ = image.shape
    foreground = np.zeros((h, w, 3))
    mask = np.zeros((h, w))

    f_startX, f_endX = max(0, x_offset), min(w, w+x_offset)
    f_startY, f_endY = max(0, y_offset), min(h, h+y_offset)
    o_startX, o_endX = max(0, -x_offset), min(w, w-x_offset)
    o_startY, o_endY = max(0, -y_offset), min(h, h-y_offset)
    foreground[f_startY:f_endY, f_startX:f_endX, :] = image[o_startY:o_endY, o_startX:o_endX, :]
    mask[f_startY:f_endY, f_startX:f_endX] = combined_mask[o_startY:o_endY, o_startX:o_endX]

    result = paste_image(foreground, image_B, mask*255)

    return result


def remove_bg(image, threshold):
    mask = removebg(image, threshold)
    cv2.imwrite("temp.png", mask*255)
    object = cut_object(image, mask)
    return object



with gr.Blocks() as remove_bg_interface:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            threshold = gr.Slider(minimum=0, maximum=255, step=1, value=10,)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")

    button.click(remove_bg, inputs=[input_image, threshold], outputs=[output_image])


with gr.Blocks() as segment_remove_interface:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")
            sliderX = gr.Slider(label="select X", minimum=0, maximum=1, step=0.01, value=0.5,)
            sliderY = gr.Slider(label="select Y", minimum=0, maximum=1, step=0.01, value=0.5,)
            n_dilate = gr.Slider(label="dilate", minimum=0, maximum=20, step=1, value=2,)
            button2 = gr.Button("Submit")
            output_object = gr.Image(type="numpy")

    button.click(image2predictor, inputs=[input_image], outputs=[output_image])
    button2.click(remove_object, inputs=[input_image, sliderX, sliderY, n_dilate], outputs=[output_image, output_object])

with gr.Blocks() as move_object_interface:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")
            sliderX = gr.Slider(label="select X", minimum=0, maximum=1, step=0.01, value=0.5,)
            sliderY = gr.Slider(label="select Y", minimum=0, maximum=1, step=0.01, value=0.5,)
            moveX = gr.Slider(label="move X", minimum=-1, maximum=1, step=0.01, value=0,)
            moveY = gr.Slider(label="move Y", minimum=-1, maximum=1, step=0.01, value=0,)
            button2 = gr.Button("Select move")

    button.click(image2predictor, inputs=[input_image], outputs=[output_image])
    button2.click(move_object, inputs=[input_image, sliderX, sliderY, moveX, moveY], outputs=[output_image])

tabbed_interface = gr.TabbedInterface(
    [remove_bg_interface, segment_remove_interface, move_object_interface],
    ["remove background", "Segment and Remove", "move object"]
)

if __name__ == "__main__":
    tabbed_interface.launch(server_port=7860 , server_name='0.0.0.0')

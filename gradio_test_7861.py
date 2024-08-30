import gradio as gr
import numpy as np

from process import *
from utils import cut_object

changeMood = ChangeMood()


def apply_effect(image, effect, intensity):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = changeMood(image, effect, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result



with gr.Blocks() as Mood_Changing:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            selected_option = gr.Radio(changeMood.effect_matrix.keys(), label="Select an effect")
            intensity = gr.Slider(label="intensity", minimum=0, maximum=1, step=0.1, value=0.5,)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")

    button.click(apply_effect, inputs=[input_image, selected_option, intensity], outputs=[output_image])


with gr.Blocks() as Cherry_Blossom:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            intensity = gr.Slider(label="intensity", minimum=0, maximum=1, step=0.1, value=0.5,)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")

    button.click(apply_cherryblossom_filters, inputs=[input_image, intensity], outputs=[output_image])


with gr.Blocks() as Sunset:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            intensity = gr.Slider(label="intensity", minimum=0, maximum=1, step=0.1, value=0.5,)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")

    button.click(apply_sunset_filters, inputs=[input_image, intensity], outputs=[output_image])

with gr.Blocks() as BackLight:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", interactive=True)
            intensity = gr.Slider(label="intensity", minimum=0, maximum=1, step=0.1, value=0.5,)
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")

    button.click(apply_backlight_filters, inputs=[input_image, intensity], outputs=[output_image])

tabbed_interface = gr.TabbedInterface(
    [Mood_Changing, Cherry_Blossom, Sunset, BackLight],
    ["change mood", "cherryblossom filter", "sunset filter", "backlight filter"]
)

if __name__ == "__main__":
    tabbed_interface.launch(server_port=7861, server_name='0.0.0.0')

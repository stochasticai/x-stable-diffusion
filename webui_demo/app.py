import gradio as gr
from model import load_model, inference
import numpy as np
import cv2


def get_styles():
    with open('static/styles.css') as f:
        styles = f.read()
        
    return styles
    
    
def inference_app(
    model_type,
    prompt_input,
    img_height,
    img_width,
    num_inference_steps,
    guidance_scale,
    seed 
):
    if model_type == "pytorch":
        model = load_model()
    
    images = inference(
        model=model,
        prompt=prompt_input,
        img_height=int(img_height),
        img_width=int(img_width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        num_images_per_prompt=1,
        seed=int(seed) if int(seed) != -1 else None
    )
    
    return np.array(images[0])
    

def create_gradio_app():    
    with gr.Blocks(css=get_styles()) as app:
        gr.HTML("""
            <p align="center">
            <img src="file/static/stochastic_logo_dark.svg" width="600" alt="Stochastic.ai"/>
            </p>

            <br>
        """)
            
        with gr.Tab("Text to image"):
            with gr.Row(equal_height=True):
                with gr.Column():                
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your input prompt. For example: A person riding a horse",
                        lines=3
                    )
                    
                with gr.Column():
                    image_output = gr.Image(
                        label="Generated image",
                        shape=[50,50]
                    )
                    
            with gr.Row(equal_height=True):
                with gr.Column():
                    model_type = gr.Dropdown(
                        label="Optimization type",
                        choices=["pytorch"],
                        value="pytorch"
                    )
                    
                    img_height = gr.Number(
                        label="Image height",
                        value=512
                    )
                    
                    img_width = gr.Number(
                        label="Image width",
                        value=512
                    )
                    
                with gr.Column():
                    num_inference_steps = gr.Slider(
                        minimum=10,
                        maximum=150,
                        value=50
                    )
                    
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        value=7.5
                    )
                    
                    seed = gr.Number(
                        label="Seed",
                        value=-1
                    )
                    
            button = gr.Button(value="Generate")
            button.click(
                fn=inference_app,
                inputs=[
                    model_type,
                    prompt_input,
                    img_height,
                    img_width,
                    num_inference_steps,
                    guidance_scale,
                    seed                    
                ],
                outputs=image_output
            )
        
    return app    
    

app = create_gradio_app() 
app.queue(concurrency_count=1)
app.launch() 

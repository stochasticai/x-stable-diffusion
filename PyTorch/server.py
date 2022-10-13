import os
from fastapi import FastAPI
from model import load_model, inference
from pydantic import BaseModel
from typing import Union, List
import torch
import numpy as np


class Item(BaseModel):
    prompt: Union[str, List[str]]
    img_height: int = 512
    img_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = None


app = FastAPI()
print("[+] Loading model")
model = load_model(
    model_name_or_path = "CompVis/stable-diffusion-v1-4" if os.getenv("MODEL_DIR_PATH") is None else os.getenv("MODEL_DIR_PATH")
)
print("[+] Model loaded")

if torch.cuda.is_available():
    print("[+] Moving the model to the GPU")
    model = model.to("cuda")
else:
    print("[+] Your model will be executed in CPU. The execution might be very slow.")


@app.post("/predict/")
async def predict(input_api: Item):
    model_input = {
        **input_api.dict(),
        **{"return_time": True}
    }
    
    images, time = inference(model=model, **model_input)
    
    images = np.array([np.array(img) for img in images]).tolist()
    
    return {
        "images": images,
        "generation_time_in_secs": time
    }
    


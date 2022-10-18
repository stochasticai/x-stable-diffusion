from fastapi import FastAPI
from model import load_model, inference
from pydantic import BaseModel
from typing import Union, List
import torch
import numpy as np
import os
from typing import Dict, Union
from PIL import Image


class Item(BaseModel):
    prompt: Union[str, List[str]]
    img_height: int = 512
    img_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = None
    

exeuction_provider = os.getenv("ONNX_EXECUTION_PROVIDER")

if exeuction_provider is None and torch.cuda.is_available():
    print("[+] Moving the model to the GPU")
    exeuction_provider="CUDAExecutionProvider"
elif exeuction_provider is None:
    print("[+] Your model will be executed in CPU. The execution might be very slow.")
    exeuction_provider="CPUExecutionProvider"
    

app = FastAPI()
print("[+] Loading model")
model = load_model(
    model_name_or_path = "CompVis/stable-diffusion-v1-4" if os.getenv("MODEL_DIR_PATH") is None else os.getenv("MODEL_DIR_PATH"),
    provider=exeuction_provider
)
print("[+] Model loaded")
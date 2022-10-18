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
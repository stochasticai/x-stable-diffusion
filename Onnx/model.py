from diffusers import StableDiffusionOnnxPipeline
import torch
from typing import List, Union
import time
from PIL import Image


def load_model(
    model_name_or_path="CompVis/stable-diffusion-v1-4",
    provider="CUDAExecutionProvider"
) -> StableDiffusionOnnxPipeline:
    """Loads the model

    :param model_name_or_path: model name or path, defaults to "CompVis/stable-diffusion-v1-4"
    :param provider: execution provider - Onnx Runtime, defaults to "CUDAExecutionProvider"
    :return: the model
    """
    
    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        model_name_or_path, 
        revision="onnx",
        provider=provider,
        use_auth_token=True,
    )
    
    return pipe
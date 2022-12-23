from diffusers import StableDiffusionOnnxPipeline
import torch
from typing import List, Union
import time
from PIL import Image


def load_model(
    model_name_or_path="CompVis/stable-diffusion-v1-4", provider="CUDAExecutionProvider"
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


def inference(
    model: StableDiffusionOnnxPipeline,
    prompt: Union[str, List[str]],
    img_height: int = 512,
    img_width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    seed: int = None,
    return_time=False,
) -> Image:
    """Function to start generating images

    :param model: model
    :param prompt: prompt
    :param img_height: image height, defaults to 512
    :param img_width: image width, defaults to 512
    :param num_inference_steps: number of inference steps, defaults to 50
    :param guidance_scale: guidance scale, defaults to 7.5
    :param num_images_per_prompt: number of images per prompt, defaults to 1
    :param seed: seed, defaults to None
    :param return_time: if the time to generate should be returned, defaults to False
    :return: the generated images and the time if return_time is True
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)

    start_time = time.time()
    output = model(
        prompt=prompt,
        height=img_height,
        width=img_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )
    end_time = time.time()

    if return_time:
        return output.images, end_time - start_time

    return output.images

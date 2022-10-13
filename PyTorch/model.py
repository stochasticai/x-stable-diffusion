from diffusers import StableDiffusionPipeline
import torch
from typing import List, Union
import time


def load_model(
    model_name_or_path="CompVis/stable-diffusion-v1-4"
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path, 
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe = pipe.to("cuda")
    
    return pipe


def inference(
    model: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    img_height: int = 512,
    img_width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    seed: int = None,
    return_time=False
):
    generator = None
    if seed is not None:
        generator = torch.Generator(device='cuda')
        generator = generator.manual_seed(seed)

    start_time = time.time()
    with torch.autocast("cuda"):
        output = model(
            prompt=prompt,
            height=img_height,
            width=img_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        )
    end_time = time.time()
    
    if return_time:
        return output.images, end_time - start_time
    
    return output.images
    
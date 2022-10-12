import torch
import argparse
import time
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci", help="input prompt")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--benchmark", action="store_true",help="Running benchmark by average num iteration")
    parser.add_argument("--n_iters", default=50, help="Running benchmark by average num iteration")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to("cuda")
    if args.benchmark:
        n_iters = args.n_iters
        #warm up
        for i in tqdm(range(3)):
            with torch.inference_mode(), torch.autocast("cuda"):
                images = pipe(args.prompt,num_inference_steps=35)
    else:
        n_ters = 1
    
    start = time.time()
    for i in tqdm(range(n_iters)):
        with torch.inference_mode(), torch.autocast("cuda"):
            images = pipe(args.prompt, num_inference_steps=35)
    end = time.time()
    if args.benchmark:
        print("Average inference time is: ",(end-start)/n_iters)
    images.images[0].save('image_generated.png')
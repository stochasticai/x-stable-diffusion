import torch
import argparse
import time
from PIL import Image
from tqdm import tqdm
from transformers import  CLIPTokenizer,CLIPTextModel

from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from torch import autocast
import gc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci", help="input prompt")
    parser.add_argument("--nvfuser_unet_save_path", default="./unet_jit.pt", type=str, help="Nvfuser unet saved path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")
    parser.add_argument("--benchmark", action="store_true",help="Running benchmark by average num iteration")
    parser.add_argument("--n_iters", default=50, help="Running benchmark by average num iteration")

    return parser.parse_args()

class NvfuserDiffusionModel():
    def __init__(self, args):
        self.device = torch.device("cuda")
        self.unet = torch.jit.load(args.nvfuser_unet_save_path)
        self.unet = self.unet.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            use_auth_token=True).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="tokenizer",
            use_auth_token=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000)

    def predict(
        self, 
        prompts,
        num_inference_steps = 50,
        height = 512,
        width = 512,
        max_seq_length = 64
    ):
        guidance_scale = 7.5
        batch_size = 1
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt")
        with torch.inference_mode(), autocast("cuda"):
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_seq_length, return_tensors="pt"
        )
        with torch.inference_mode(), autocast("cuda"):
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8)).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.sigmas[0]
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            gc.collect()
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            noise_pred = self.unet(latent_model_input, torch.tensor([t]).to(self.device), text_embeddings)[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            with torch.inference_mode(), autocast("cuda"):
                latents = self.scheduler.step(noise_pred, i, latents)["prev_sample"]


        # # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.inference_mode(), autocast("cuda"):
            image = self.vae.decode(latents).sample
        return image

if __name__ == "__main__":
    args = get_args()
    model = NvfuserDiffusionModel(args)
    if args.benchmark:
        n_iters = args.n_iters
        #warm up
        for i in range(3):
            image = model.predict(
                prompts = args.prompt,
                num_inference_steps = 35,
                height = args.img_size[0],
                width = args.img_size[1],
                max_seq_length = args.max_seq_length
            )
    else:
        n_iters = 1

    start = time.time()
    for i in tqdm(range(n_iters)):
        image = model.predict(
            prompts = args.prompt,
            num_inference_steps = 35,
            height = args.img_size[0],
            width = args.img_size[1],
            max_seq_length = args.max_seq_length
        )
    end = time.time()
    if args.benchmark:
        print("Average inference time is: ",(end-start)/n_iters)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save('image_generated.png')
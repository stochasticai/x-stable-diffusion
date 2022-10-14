import torch
import argparse
from diffusers import UNet2DConditionModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",default="./unet_jit.pt", type=str, help="Nvfuser saved path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")

    return parser.parse_args()

def convert(args):
    device = torch.device("cuda")
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_auth_token=True).to(device)
    unet.eval()

    latents = torch.randn((args.batch_size, 4, args.img_size[0] // 8, args.img_size[1] // 8))
    latent_model_input = torch.cat([latents] * 2).to(device)
    text_embeddings = torch.randn((args.batch_size, args.max_seq_length,768)).float().to(device)
    text_embeddings = torch.cat([text_embeddings,text_embeddings])
    timestep_ = torch.tensor([10]).to(device)
    with torch.no_grad():
        with torch.autocast("cuda"):
            traced_applymodel_half = torch.jit.trace(unet,
                (latent_model_input, timestep_, text_embeddings),check_trace = False
            )

    traced_applymodel_half.save(args.save_path)

if __name__ == "__main__":
    args = get_args()
    convert(args)
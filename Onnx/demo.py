import argparse
from model import load_model, inference
from pathlib import Path
import uuid


def get_args():
    """Configure the CLI arguments

    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci", help="input prompt")
    parser.add_argument("--provider", default="CUDAExecutionProvider", help="ONNX Execution provider")
    parser.add_argument("--img_height", type=int, default=512, help="The height in pixels of the generated image.")
    parser.add_argument("--img_width", type=int, default=512, help="The width in pixels of the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to make generation deterministic")
    parser.add_argument("--saving_path", type=str, default="generated_images", help="Directory where the generated images will be saved")
    

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Create directory to save images if it does not exist
    saving_path = Path(args.saving_path)
    if not saving_path.exists():
        saving_path.mkdir(exist_ok=True, parents=True)
    
    print("[+] Loading the model")
    model = load_model()
    print("[+] Model loaded")
    
    print("[+] Generating images...")
    # PIL images
    images, time = inference(
        model=model,
        prompt=args.prompt,
        img_height=args.img_height,
        img_width=args.img_width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        return_time=True
    )
    
    print("[+] Time needed to generate the images: {} seconds".format(time))
    
    # Save PIL images with a random name
    for img in images:
        img.save('{}/{}.png'.format(
            saving_path.as_posix(),
            uuid.uuid4()
        ))
            
    print("[+] Images saved in the following path: {}".format(saving_path.as_posix()))
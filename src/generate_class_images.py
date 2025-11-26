import argparse
import os
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

def generate_class_images(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pipeline
    # Use float16 for GPU to save memory, float32 for CPU
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading model: {args.model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None # Optional: disable safety checker to save memory/speed if safe
    )
    pipe.to(device)

    # Enable memory optimizations if on GPU
    if device == "cuda":
        pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention() # Uncomment if xformers is installed

    print(f"Generating {args.num_images} images for class prompt: '{args.prompt}'")
    
    # Check how many images already exist to avoid overwriting or re-generating if resumed
    existing_images = [f for f in os.listdir(args.output_dir) if f.endswith(".png") or f.endswith(".jpg")]
    start_idx = len(existing_images)
    
    if start_idx >= args.num_images:
        print(f"Found {start_idx} images already. Skipping generation.")
        return

    # Generate images
    for i in tqdm(range(start_idx, args.num_images)):
        image = pipe(args.prompt).images[0]
        
        filename = f"{args.output_dir}/{args.starting_id + i:04d}.png"
        image.save(filename)

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class images for DreamBooth prior preservation.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID")
    parser.add_argument("--prompt", type=str, default="a cat", help="Class prompt (e.g., 'a dog', 'a person')")
    parser.add_argument("--num_images", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="dataset/class_images", help="Output directory for images")
    parser.add_argument("--starting_id", type=int, default=0, help="Starting ID for image filenames")
    args = parser.parse_args()
    generate_class_images(args)

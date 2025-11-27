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
    
    # Generate images
    print(f"Generating {args.num_images} images starting from ID {args.starting_id} in batches of {args.batch_size}...")
    
    for i in tqdm(range(0, args.num_images, args.batch_size)):
        # Calculate actual batch size for the last batch
        current_batch_size = min(args.batch_size, args.num_images - i)
        
        # Create lists of prompts
        prompts = [args.prompt] * current_batch_size
        negative_prompts = [args.negative_prompt] * current_batch_size if args.negative_prompt else None
        
        # Generate batch
        images = pipe(prompts,
                     negative_prompt=negative_prompts,
                     num_inference_steps=50,
                     guidance_scale=6.0,
                     generator=torch.manual_seed(42 + i)).images # Seed changed to vary per batch
        
        # Save images
        for j, image in enumerate(images):
            image_id = args.starting_id + i + j
            filename = f"{args.output_dir}/{image_id:04d}.png"
            image.save(filename)

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class images for DreamBooth prior preservation.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID")
    parser.add_argument("--prompt", type=str, default="a cat", help="Class prompt (e.g., 'a dog', 'a person')")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to avoid specific features")
    parser.add_argument("--num_images", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="dataset/class_images", help="Output directory for images")
    parser.add_argument("--starting_id", type=lambda x: int(float(x)), default=0, help="Starting ID for image filenames")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    args = parser.parse_args()
    generate_class_images(args)

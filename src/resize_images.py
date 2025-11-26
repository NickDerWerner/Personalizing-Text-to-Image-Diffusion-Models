from PIL import Image
import os
import argparse
from tqdm import tqdm

def resize_images(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    print(f"Found {len(input_files)} images in {args.input_dir}")
    
    for filename in tqdm(input_files):
        img_path = os.path.join(args.input_dir, filename)
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize/Crop logic
                # Center crop to square then resize
                width, height = img.size
                new_dim = min(width, height)
                
                left = (width - new_dim) / 2
                top = (height - new_dim) / 2
                right = (width + new_dim) / 2
                bottom = (height + new_dim) / 2
                
                img = img.crop((left, top, right, bottom))
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
                
                output_path = os.path.join(args.output_dir, filename)
                img.save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Resizing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and crop instance images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw instance images")
    parser.add_argument("--output_dir", type=str, default="dataset/instance_images", help="Directory to save resized images")
    parser.add_argument("--size", type=int, default=512, help="Target resolution (e.g., 512 or 768)")
    
    args = parser.parse_args()
    resize_images(args)

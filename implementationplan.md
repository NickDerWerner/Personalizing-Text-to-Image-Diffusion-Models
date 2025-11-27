
## Description

Students will explore methods like **DreamBooth [[**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**](https://www.notion.so/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation-1f37d0e6cb8980b6b312f52d1d6b4aee?pvs=21)]** and **Custom Diffusion [[**Multi-Concept Customization of Text-to-Image Diffusion**](https://www.notion.so/Multi-Concept-Customization-of-Text-to-Image-Diffusion-1f37d0e6cb898033813fe5252fc8897a?pvs=21)]**, which fine-tune large diffusion models on as few as 3–5 reference images of a new “subject.” The project will implement a pipeline where a unique text token is assigned to the new concept and the diffusion model is fine-tuned such that it binds that token to the subject’s visual features. They will experiment with loss functions that preserve the model’s prior diversity (e.g. prior preservation loss) to prevent overfitting.


This implementation plan is designed for a student team using Google Colab (for shared compute), Google Drive (for storage), and GitHub (for code versioning).

Since DreamBooth fine-tuning produces large model files (2–4 GB per checkpoint), managing storage between Colab and Drive is critical.

Phase 1: Environment & Infrastructure

Goal: Set up a reproducible development environment that connects your code (GitHub), data/weights (Drive), and compute (Colab/Local).

    Repository Structure (GitHub): Create a repo named dreambooth-implementation.

        src/: Contains your training scripts.

        notebooks/: Contains the Colab notebooks for running training and inference.

        requirements.txt: List dependencies (diffusers, transformers, accelerate, torch, torchvision, bitsandbytes).

        .gitignore: Crucial. Add *.ckpt, *.safetensors, and output_dirs/ to ignore large model weights.

    Colab Setup:

        Mount Google Drive: You must mount Drive to store the fine-tuned model; otherwise, it will vanish when the Colab runtime disconnects.

        GPU Selection: Ensure you are using a GPU runtime (T4 is standard on free Colab; A100 is ideal if you have Pro). DreamBooth requires ~12GB VRAM without optimizations, or ~8GB with 8-bit Adam and gradient_checkpointing.

Phase 2: Data Preparation Pipeline

Goal: Automate the creation of the training dataset and the "regularization" dataset (for Prior Preservation).

    Instance Data (Your Subject):

        Create a folder dataset/instance_images.

        Upload your 3–5 images here.

        Task: Write a simple script to resize/crop these to 512x512 (or 768x768 for SDXL) to avoid runtime resizing artifacts.

    Class Data (Prior Preservation):

        Concept: Before training, you need ~200 images of the generic class (e.g., "a cat") to serve as the regularization set.

        Implementation: Create a script generate_class_images.py.

            Load the base model (e.g., runwayml/stable-diffusion-v1-5).

            Loop 200 times generating images with the prompt "A dog" (or whatever your class is).

            Save these to dataset/class_images on Drive.

Phase 3: The Training Loop (Implementation)

Goal: Write the actual fine-tuning logic. You will likely use the Hugging Face diffusers library as a base, but you should understand the components.

Key Components to Implement:

    The Dataset Class:

        You need a custom PyTorch Dataset class that returns a dictionary containing:

            instance_images: The pixel values of your subject.

            instance_prompt_ids: Tokenized version of "A [V] dog".

            class_images: The pixel values of the generic dog (from Phase 2).

            class_prompt_ids: Tokenized version of "A dog".

    The Model Architecture:

        Load the VAE (Variational Autoencoder), Tokenizer, and Text Encoder (usually frozen/not trained).

        Load the UNet (this is what you train).

        Optimization: Enable "Gradient Checkpointing" on the UNet to save memory.

    The Loss Function (The Math Part):

        Standard DreamBooth uses MSE (Mean Squared Error) loss.

        Logic:

            Pass Instance Images → Calculate noise prediction error (Loss A).

            Pass Class Images → Calculate noise prediction error (Loss B).

            Total Loss = Loss A + (λ× Loss B).

        Note: usually λ (prior loss weight) is set to 1.0.

    The Optimizer:

        Use bitsandbytes.optim.AdamW8bit. This is the "secret sauce" to fitting DreamBooth on consumer GPUs (like the free Colab T4). It reduces optimizer state memory by nearly 75%.

Phase 4: Training & Checkpointing

Goal: Run the training and save the result safely.

    Run Configuration:

        Learning Rate: Start with 5×10−6.

        Steps: 800–1000 steps is usually the sweet spot for 5 instance images.

        Batch Size: 1 (due to VRAM limits).

    Saving the Model:

        Don't just save the .ckpt file. You need to save the entire pipeline so it's easy to load later.

        Use pipeline.save_pretrained("drive/MyDrive/my_dreambooth_model").

Phase 5: Inference (Testing)

Goal: Generate new images using your fine-tuned model.

    Create a separate notebook: inference.ipynb.

    Load the model path from Google Drive.

    Prompt Engineering:

        Test the token binding: "A [V] dog in a bucket."

        Test prior preservation: "A dog running" (should look like a generic dog, not your specific one).
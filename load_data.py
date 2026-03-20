import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

def save_dataset_to_folders(dataset, root_path, split="train"):
    """
    Converts a Hugging Face dataset into a local directory structure
    compatible with torchvision.datasets.ImageFolder.
    Structure: root_path/class_name/image_001.jpg
    """
    os.makedirs(root_path, exist_ok=True)

    # Get the human-readable class names from the dataset features
    # (If the dataset uses integer IDs for labels)
    class_names = dataset[split].features["label"].names

    print(f"Saving {split} split to {root_path}...")
    for i, item in enumerate(tqdm(dataset[split])):
        image = item["image"]
        label_id = item["label"]
        class_name = class_names[label_id].replace(" ", "_")

        # Create class directory
        class_dir = os.path.join(root_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert to RGB if necessary (to avoid issues with PNG/RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save image
        image_path = os.path.join(class_dir, f"img_{i:06d}.jpg")
        image.save(image_path, "JPEG", quality=95)

if __name__ == "__main__":
    # Base data directory
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)

    # ==========================================
    # Phase 1: ImageNet-100 (REPA Pre-training)
    # ==========================================
    print("\n--- Downloading ImageNet-100 ---")
    # Using a popular community subset of ImageNet
    ds_imagenet = load_dataset("clane9/imagenet-100")

    imagenet_path = os.path.join(base_dir, "imagenet")
    save_dataset_to_folders(ds_imagenet, imagenet_path, split="train")

    # ==========================================
    # Phase 2: ArtBench-10 (LoRA Fine-tuning)
    # ==========================================
    print("\n--- Downloading ArtBench-10 ---")
    # ArtBench-10 contains 60,000 images across 10 art styles
    ds_artbench = load_dataset("artbench/artbench-10")

    artbench_path = os.path.join(base_dir, "my_style_data")
    save_dataset_to_folders(ds_artbench, artbench_path, split="train")

    print("\n[SUCCESS] Datasets are ready in the /data folder.")
    print(f"REPA Data: {imagenet_path}")
    print(f"LoRA Data: {artbench_path}")
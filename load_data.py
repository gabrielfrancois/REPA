import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import re

def save_dataset_to_folders(dataset, root_path, split="train", dataset_name="imagenet"):
    """
    Converts a Hugging Face dataset into a local directory structure
    compatible with torchvision.datasets.ImageFolder.
    """
    os.makedirs(root_path, exist_ok=True)

    if dataset_name == "artbench":
        # ArtBench: extract class names from prompts SAFELY
        prompts_sample = dataset[split]["prompt"][:100]
        class_names = set()
        
        for p in prompts_sample:
            match = re.search(r"a (\w+(?:\s+\w+)*?) painting", p.lower())
            if match:
                class_name = match.group(1).replace(" ", "_")
                class_names.add(class_name)
        
        class_names = sorted(class_names)
        print(f"ArtBench classes: {len(class_names)} ({class_names[:3]}...)")

        def get_class_name(prompt):
            match = re.search(r"a (\w+(?:\s+\w+)*?) painting", prompt.lower())
            return match.group(1).replace(" ", "_") if match else "unknown"
        
    else:  # ImageNet
        class_names = dataset[split].features["label"].names
        def get_class_name(label_id):
            return class_names[label_id].replace(" ", "_")

    print(f"Saving {split} split to {root_path} ({len(class_names)} classes)...")
    for i, item in enumerate(tqdm(dataset[split])):
        image = item["image"]
        
        if dataset_name == "artbench":
            label_str = get_class_name(item["prompt"])
            if label_str == "unknown":
                continue  # Skip problematic items
        else:
            label_id = item["label"]
            label_str = get_class_name(label_id)

        class_dir = os.path.join(root_path, label_str)
        os.makedirs(class_dir, exist_ok=True)

        if image.mode != "RGB":
            image = image.convert("RGB")

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
    ds_imagenet = load_dataset("clane9/imagenet-100")

    imagenet_path = os.path.join(base_dir, "imagenet")
    save_dataset_to_folders(ds_imagenet, imagenet_path, split="train", dataset_name="imagenet")

    # ==========================================
    # Phase 2: ArtBench-10 (LoRA Fine-tuning)
    # ==========================================
    print("\n--- Downloading ArtBench-10 ---")
    ds_artbench = load_dataset("Doub7e/ArtBench-10")

    artbench_path = os.path.join(base_dir, "my_style_data")
    save_dataset_to_folders(ds_artbench, artbench_path, split="train", dataset_name="artbench")

    print("\n[SUCCESS] Datasets are ready in the /data folder.")
    print(f"REPA Data: {imagenet_path}")
    print(f"LoRA Data: {artbench_path}")

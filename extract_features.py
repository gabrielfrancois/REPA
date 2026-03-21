import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import timm
from PIL import Image
import numpy as np
from tqdm import tqdm

# Same crop function from train.py
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 256
    data_path = "data/imagenet"
    save_path = "data/imagenet_features"

    print("Loading Models...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    teacher = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True).to(device).eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    os.makedirs(save_path, exist_ok=True)
    for class_name in dataset.classes:
        os.makedirs(os.path.join(save_path, class_name), exist_ok=True)

    print("Extracting features...")
    img_idx = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        
        # Get Teacher Features
        x_teacher = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        teacher_out = teacher.forward_features(x_teacher)[:, 1:] # (B, 256, 768)
        
        # Get Latents
        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215) # (B, 4, 32, 32)

        # save to disk (move to CPU to save)
        teacher_out = teacher_out.cpu()
        x_latent = x_latent.cpu()
        y = y.cpu()

        for i in range(x.shape[0]):
            class_name = dataset.classes[y[i].item()]
            save_file = os.path.join(save_path, class_name, f"img_{img_idx:06d}.pt")
            
            torch.save({
                "latent": x_latent[i].clone(),
                "repa": teacher_out[i].clone()
            }, save_file)
            img_idx += 1

if __name__ == "__main__":
    main()
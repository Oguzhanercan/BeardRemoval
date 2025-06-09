import subprocess
import os
import os
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import requests
from dataset.MagicQuill import folder_paths
from dataset.MagicQuill.scribble_color_edit import ScribbleColorEditModel
import time
import io
from tqdm import tqdm
import cv2



def tensor_to_image(tensor):
    tensor = tensor.squeeze(0) * 255.
    pil_image = Image.fromarray(tensor.cpu().byte().numpy())
    
    return pil_image

def read_base64_image(base64_image):
    if base64_image.startswith("data:image/png;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/jpeg;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/webp;base64,"):
        base64_image = base64_image.split(",")[1]
    else:
        raise ValueError("Unsupported image format.")
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image)
    return image

def create_alpha_mask(base64_image):
    """Create an alpha mask from the alpha channel of an image."""
    image = Image.open(base64_image)
    image = image.resize((512,512))
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
    alpha_channel = np.where(np.array(image).astype(np.float32)  < 50, 0, 255) / 255.0
    mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask

def load_and_preprocess_image(base64_image, convert_to='RGB', has_alpha=False):
    """Load and preprocess a base64 image."""
    image = Image.open(base64_image)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def load_and_resize_image(im_path, convert_to='RGB', max_size=512):
    """Load and preprocess a base64 image, resize if necessary."""
    Image.open(im_path)
    width, height = image.size
    scaling_factor = max_size / min(width, height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def prepare_images_and_masks(total_mask, original_image, add_color_image = None, add_edge_image = None, remove_edge_image = None):
    total_mask = create_alpha_mask(total_mask)
    original_image_tensor = load_and_preprocess_image(original_image)
    if add_color_image:
        add_color_image_tensor = load_and_preprocess_image(add_color_image)
    else:
        add_color_image_tensor = original_image_tensor
    
    add_edge_mask = create_alpha_mask(add_edge_image) if add_edge_image else torch.zeros_like(total_mask)
    remove_edge_mask = create_alpha_mask(remove_edge_image) if remove_edge_image else torch.zeros_like(total_mask)
    return add_color_image_tensor, original_image_tensor, total_mask, add_edge_mask, remove_edge_mask


def generate(scribbleColorEditModel, total_mask, 
            original_image,
            positive_prompt,
            negative_prompt = "",
            ckpt_name = os.path.join('SD1.5', 'realisticVisionV60B1_v51VAE.safetensors'),
            grow_size = 15, 
            stroke_as_edge = "enable",  # TRUE
            fine_edge = "disable",
            edge_strength = 0.55, 
            color_strength = 0.55,  
            inpaint_strength = 1, 
            seed = 42, 
            steps = 20, 
            cfg = 5, 
            sampler_name = "euler_ancestral", 
            scheduler = "karras",
            progress = None):
    add_color_image, original_image, total_mask, add_edge_mask, remove_edge_mask = prepare_images_and_masks(total_mask, original_image, None, total_mask, None)
    
    total_mask = torch.where(total_mask == 0, torch.tensor(1), 0)
    add_edge_mask = torch.where(add_edge_mask == 0, torch.tensor(1), 0)
    remove_edge_mask  = torch.where(remove_edge_mask == 0, torch.tensor(1), 0)

    """print(np.unique(total_mask))
   
    print("add color", add_color_image)
    print(total_mask)
    print("add edge", add_edge_mask)
    print("remove edge", remove_edge_mask)


    cv2.imwrite("remove_edge_mask_2.png",(np.array(torch.squeeze(remove_edge_mask))*255).astype(np.uint8))
    cv2.imwrite("addege_2.png",(np.array(torch.squeeze(add_edge_mask))*255).astype(np.uint8))
    cv2.imwrite("color_image2.png",(np.array(torch.squeeze(add_color_image))*255).astype(np.uint8))
    cv2.imwrite("totalmask2.png",(np.array(torch.squeeze(total_mask))*255).astype(np.uint8))"""


    if torch.sum(remove_edge_mask).item() > 0 and torch.sum(add_edge_mask).item() == 0:
        if positive_prompt == "":
            positive_prompt = "empty scene"
        edge_strength /= 3.

    latent_samples, final_image, lineart_output, color_output = scribbleColorEditModel.process(
        ckpt_name,
        original_image, 
        add_color_image, 
        positive_prompt, 
        negative_prompt, 
        total_mask, 
        total_mask, 
        remove_edge_mask, 
        grow_size, 
        stroke_as_edge, 
        fine_edge,
        edge_strength, 
        color_strength,  
        inpaint_strength, 
        seed, 
        steps, 
        cfg, 
        sampler_name, 
        scheduler,
        progress
    )
    return final_image



#base model path ,
# fine edge disable
# grow size = 15              0 -- 100
# edge_strength = 0.55 0 - 5
#color strenght = 0 - 5 0.55
#inpaint strgenth = 0 -5 1
# cfg = 5    
# sampler = euler_ancestral
# scheduler = karras
    
#


class ImageEdit():
    def __init__(self,masks,images):


        self.scribbleColorEditModel = ScribbleColorEditModel()



        self.mask_prefix = masks
        self.img_prefix= images
        self.mask_list = os.listdir(self.mask_prefix) 

        os.makedirs("dataset/dataset/CleanFaces",exist_ok = True)


    def forward(self,limit):
        for mask in tqdm(self.mask_list[:limit]):
            mask_path = os.path.join(self.mask_prefix,mask)
            #mask_img = Image.open(mask_path)
            #mask_img.save("mask.png")
            img_path = os.path.join(self.img_prefix,mask)
            #img = Image.open(img_path)
            #img.save("img.png")
            result = generate(self.scribbleColorEditModel,mask_path,img_path,positive_prompt="clean shaved face, no beard, no facial hair")
            result= tensor_to_image(result)
            result.save(f"dataset/dataset/CleanFaces/{mask}")



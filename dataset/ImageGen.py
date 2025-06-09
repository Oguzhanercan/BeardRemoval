import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
import numpy as np
import cv2
import os




class ImageGenerator():
    def __init__(self,model_id,nf4_id):

        self.model_nf4 = FluxTransformer2DModel.from_pretrained(nf4_id, torch_dtype=torch.bfloat16)


        self.pipe = FluxPipeline.from_pretrained(model_id, transformer=self.model_nf4, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()


        with open("dataset/dataset/prompts.txt","r") as prompts:
            lines = prompts.readlines()
            print(lines)
        self.lines = lines
        self.prompt_prefix = "Realistic, real life photo, face photo of person, face centered of the image,  "

        os.makedirs("dataset/dataset/BeardFaces",exist_ok=True)


    def forward(self,limit = None): # limit is the number of images to generate.
        if limit == None:
            limit = len(self.prompt_prefix)
        elif not isinstance(limit,int):
            print("limit is not a valid number, it should be integer.")
            exit()

        for c,line in enumerate(self.lines[:limit]):

            image = self.pipe(
            prompt = self.prompt_prefix + line
            , num_inference_steps       = 20
            , guidance_scale            = 3.5,
            width = 512,
            height = 512
        ).images[0]

            image.save(f"dataset/dataset/BeardFaces/example{str(c)}.png")
from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm
# Load a model


class Segment():
    def __init__(self, model_path,path2images = "dataset/dataset/BeardFaces"):

        self.model = YOLO(model_path).to("cuda")  # load an official model

        self.path2images = path2images

        self.images = os.listdir(path2images)

    os.makedirs("dataset/dataset/segmentation_results",exist_ok=True)
    os.makedirs("dataset/dataset/masks",exist_ok=True)
    def forward(self,limit): 

        ok_masks = 0
        failed_detections = 0
        for c,img_path in tqdm(enumerate(self.images[:limit])):
            
            # Predict with the model
            try:
                results = self.model(os.path.join(self.path2images,img_path))  # predict on an image

                for counter, detection in enumerate(results[0].masks.data):
                    
                    
                    cls_id = int(results[0].boxes[counter].cls.item())
                        
                    cls_name = self.model.names[cls_id]
                    if cls_name == "beard" and results[0].boxes[counter].conf >= 0.80:
                        ok_masks +=1 
                        result = results[0]
                        
                        masks = result.masks  # Masks object for segmentation masks outputs
                        mask = masks[counter].data.squeeze().cpu().numpy() * 255  
                        mask = mask.astype(np.uint8) # Convert mask to uint8 if needed

                        cv2.imwrite(f'dataset/dataset/masks/{img_path}', mask)  

                    
                        
                        obb = result.obb  # Oriented boxes object for OBB outputs
                        result.save(filename=f"dataset/dataset/segmentation_results/{img_path}")  # save to disk

                        continue
            

            except:
                failed_detections+=1

        print(f"Completed, detection failed for {str(len(self.images[:limit]) - ok_masks)} examples. ")
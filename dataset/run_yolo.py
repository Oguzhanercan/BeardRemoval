from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm
# Load a model
model = YOLO("/home/oguzhan/Desktop/AppNationCase/Beard-Hair-Image-Segmentation/models/best_hair_117_epoch_v4.pt").to("cuda")  # load an official model

path2images = "/home/oguzhan/Desktop/AppNationCase/male2/"

images = os.listdir(path2images)

os.makedirs("./segmentation_results2/images",exist_ok=True)
os.makedirs("./segmentation_results2/masks",exist_ok=True)
failed_detections = 0
for c,img_path in tqdm(enumerate(images)):
    print(c)
    # Predict with the model
    try:
        results = model(os.path.join(path2images,img_path))  # predict on an image

        for counter, detection in enumerate(results[0].masks.data):
            
            
            cls_id = int(results[0].boxes[counter].cls.item())
                
            cls_name = model.names[cls_id]
            if cls_name == "beard" and results[0].boxes[counter].conf >= 0.80:
                result = results[0]
                
                masks = result.masks  # Masks object for segmentation masks outputs
                mask = masks[counter].data.squeeze().cpu().numpy() * 255  
                mask = mask.astype(np.uint8) # Convert mask to uint8 if needed

                cv2.imwrite(f'segmentation_results2/masks/{img_path}', mask)  

              
                
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.save(filename=f"segmentation_results2/images/{img_path}")  # save to disk

                continue
     

    except:
        failed_detections+=1

print(f"Completed, detection failed for {str(failed_detections)} examples. ")
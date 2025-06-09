import os
import matplotlib.pyplot as plt
from PIL import Image

def show_image_gird(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files = image_files[:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  

    for ax, img_file in zip(axes.flat, image_files):
     
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        
      
        ax.imshow(img)
        
        ax.axis('off') 

 
    for ax in axes.flat[len(image_files):]:
        ax.axis('off')
    
    plt.show()


def display_images_side_by_side(folder1_path, folder2_path, titles=("Folder 1", "Folder 2")):
    folder1_files = [f for f in os.listdir(folder1_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    n = 10#len(folder1_files)

    fig, axes = plt.subplots(2, n, figsize=(n * 5, 10))  
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for col, img_file in enumerate(folder1_files[:n]):
        img_path = os.path.join(folder1_path, img_file)
        img = Image.open(img_path)
        axes[0, col].imshow(img)
        axes[0, col].set_title(titles[0] if col == 0 else "")
        axes[0, col].axis('off')  

    for col, img_file in enumerate(folder1_files[:n]):
        img_path = os.path.join(folder2_path, img_file)
        img = Image.open(img_path)
        axes[1, col].imshow(img)
        axes[1, col].set_title(titles[1] if col == 0 else "")
        axes[1, col].axis('off')


    plt.show()
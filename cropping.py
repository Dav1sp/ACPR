import os
from PIL import Image, ImageOps
from torchvision import transforms
from facenet_pytorch import MTCNN
import torch
import torchvision.transforms.functional as F


INPUT_DIR = './images'               
OUTPUT_DIR = './cropped_images'     
IMAGE_SIZE = 128                     

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=0, keep_all=False, device=device)

to_pil = transforms.ToPILImage()

for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    out_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        out_path = os.path.join(out_class_path, img_name)

        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                face = face.clamp(0, 1)
                face_img = F.to_pil_image(face)
                face_img = ImageOps.autocontrast(face_img)
                face_img.save(out_path)
                print(f"Saved: {out_path}")
            else:
                print(f"No face found in {img_name}, skipping.")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
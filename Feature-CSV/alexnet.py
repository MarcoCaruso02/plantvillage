import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pretrained AlexNet and keep only the first conv layer + ReLU
alexnet = models.alexnet(pretrained=True).features[:2]
alexnet.eval()

# Preprocessing pipeline for AlexNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to extract features from the first 48 conv filters
def extract_alexnet_conv1_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(img_rgb).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = alexnet(input_tensor)[0]  # Shape: (64, H, W)

    features = []
    for i in range(48):  # First 48 filters
        fmap = output[i].numpy()
        features.append(fmap.mean())
        features.append(fmap.var())
    return features

# Dataset settings
folder_name = 'segmented_denoised/'
desease_names = [
    'Bacterial_spot', 'Healthy', 'Mold', 'Septoria_Spot',
    'Spider_mites', 'Target_Spot', 'Mosaic_virus', 'Yellow_Curl_Virus'
]

data = []
labels = []

# Loop through each class folder and image
for label in desease_names:
    class_folder = os.path.join(folder_name, label)
    if not os.path.isdir(class_folder):
        print(f"Skipping missing folder: {class_folder}")
        continue

    for filename in os.listdir(class_folder):
        img_path = os.path.join(class_folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipped unreadable image: {img_path}")
                continue

            features = extract_alexnet_conv1_features(img)
            data.append(features)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Feature names and dataframe creation
columns = [f'alexnet_f{i}_mean' for i in range(48)] + \
          [f'alexnet_f{i}_var' for i in range(48)]

df = pd.DataFrame(data, columns=columns)
df['label'] = labels

# Save features to CSV
df.to_csv('features_AlexNet_conv1_48.csv', index=False)
print("Feature extraction complete. Saved to 'features_AlexNet_conv1_48.csv'.")

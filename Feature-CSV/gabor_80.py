import numpy as np
import os
import cv2
import pandas as pd
from skimage.filters import gabor

def extract_gabor_features(image, frequencies, thetas):
    features = []
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    for theta in thetas:
        for freq in frequencies:
            filt_real, filt_imag = gabor(image, frequency=freq, theta=theta)
            magnitude = np.sqrt(filt_real**2 + filt_imag**2)
            features.append(magnitude.mean())
            features.append(magnitude.var())
    return features

# Parameters
frequencies = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.4, 0.5, 0.6,0.7 ]  # 8 frequencies
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]    # 4 orientations (0 to 157.5 degrees)

folder_name = 'segmented/'
disease_names = ['Bacterial_spot', 'Healthy', 'Mold', 'Septoria_Spot',
                 'Spider_mites', 'Target_Spot', 'Mosaic_virus', 'Yellow_Curl_Virus']

data = []
labels = []
columns = []

total_classes = len(disease_names)

for class_index, label in enumerate(disease_names, 1):
    class_dir = os.path.join(folder_name, label)
    try:
        filenames = os.listdir(class_dir)
    except Exception as e:
        print(f"[{class_index}/{total_classes}] Could not read directory {class_dir}: {e}")
        continue

    total_files = len(filenames)
    print(f"[{class_index}/{total_classes}] Processing class: {label} ({total_files} images)")

    for idx, filename in enumerate(filenames, 1):
        file_path = os.path.join(class_dir, filename)
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"  [{idx}/{total_files}] Failed to read image: {file_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            features = extract_gabor_features(gray, frequencies, thetas)
            data.append(features)
            labels.append(label)

            if idx % 10 == 0 or idx == total_files:
                print(f"  [{idx}/{total_files}] Processed {filename}")

        except Exception as e:
            print(f"  [{idx}/{total_files}] Error processing {file_path}: {e}")

# Create column names
for theta in thetas:
    for freq in frequencies:
        columns.append(f'gabor_mean_f{freq:.2f}_t{theta:.2f}')
        columns.append(f'gabor_var_f{freq:.2f}_t{theta:.2f}')

df = pd.DataFrame(data, columns=columns)
df['label'] = labels
csv_path = "features_Gabor_v6.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ Gabor feature extraction completed. CSV saved to: {csv_path}")

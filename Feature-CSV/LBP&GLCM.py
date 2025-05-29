import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

def extract_glcm_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_ubyte(image)
    glcm = graycomatrix(image, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    features = []
    for prop in props:
        values = graycoprops(glcm, prop)
        mean_over_angles = values.mean(axis=1)
        features.extend(mean_over_angles)
    return np.array(features)

def histogramLBP(P, R, image):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Parametri
folder_name = 'segmented_denoised/'
desease_name = [
    'Bacterial_spot', 'Healthy', 'Mold', 'Septoria_Spot',
    'Spider_mites', 'Target_Spot', 'Mosaic_virus', 'Yellow_Curl_Virus'
]

# Scegli modalità
print("Scegli modalità estrazione feature:")
print("1 = LBP20 (8LBP su H e S)")
print("2 = LBP84 (8, 12, 16 LBP su H e S)")
print("3 = LBP84 + GLCM18")
mode = int(input("Inserisci modalità (1, 2 o 3): "))

data = []
labels = []
columns = []

for label in desease_name:
    path = os.path.join(folder_name, label)
    print(f"Processing: {path}")

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Immagine non caricata correttamente")

            image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image_hsv)

            feature_vector = []
            feature_names = []

            # --- LBP 8 su H e S (comune a tutte le modalità) ---
            P, R = 8, 1
            n_bins = P + 2
            hist8LBPh = histogramLBP(P, R, h)
            hist8LBPs = histogramLBP(P, R, s)
            feature_vector.extend(hist8LBPh)
            feature_vector.extend(hist8LBPs)
            feature_names.extend([f"8LBP{i}h" for i in range(n_bins)])
            feature_names.extend([f"8LBP{i}s" for i in range(n_bins)])

            # --- LBP 12 e 16 (solo modalità 2 e 3) ---
            if mode >= 2:
                for P, R, ch, ch_name in [(12, 2, h, 'h'), (12, 2, s, 's'),
                                          (16, 3, h, 'h'), (16, 3, s, 's')]:
                    n_bins = P + 2
                    hist = histogramLBP(P, R, ch)
                    feature_vector.extend(hist)
                    feature_names.extend([f"{P}LBP{i}{ch_name}" for i in range(n_bins)])

            # --- GLCM (solo modalità 3) ---
            if mode == 3:
                glcm_features = extract_glcm_features(img)
                feature_vector.extend(glcm_features)
                feature_names.extend([f'{prop}_d{d}' for prop in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'] for d in [1, 2, 3]])

            data.append(feature_vector)
            labels.append(label)

            if not columns:
                columns = feature_names

        except Exception as e:
            print(f"Errore con {file_path}: {e}")

# Salva CSV
df = pd.DataFrame(data, columns=columns)
df['label'] = labels
name="LBP_max_GLCM(D2_2).csv"
df.to_csv(name, index=False)
print("CSV salvato.csv")

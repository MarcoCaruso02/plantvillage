import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import skimage
import tqdm
import skimage as ski
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from skimage.transform import resize
import cv2


folder_name='segmented/'
desease_name=['Bacterial_spot',
              'Healthy',
              'Mold',
              'Septoria_Spot',
              'Spider_mites',
              'Target_Spot',
              'Mosaic_virus',
              'Yellow_Curl_Virus']


#LBP Parameters
R=1
P=8
method='uniform'


data = []
labels = []
# Numero di possibili pattern per 'uniform', LBP
n_bins = P + 2
#new_folder_name=folder_name+desease_name[0]
for label in desease_name:
    new_folder_name = folder_name + label

    os.listdir(new_folder_name)
    print(os.listdir(new_folder_name))

    for filename in os.listdir(new_folder_name):
      file_path=os.path.join(new_folder_name,filename)
      try:
        #img=imread(file_path)
        img = cv2.imread(file_path)


        #LBP extraction
        #Convert into hsv
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)

        #LBP on h
        #LBP Uniform-two rotation invariant
        LBPresultH=skimage.feature.local_binary_pattern(h, P, R, method='uniform')
        LBPresultS=skimage.feature.local_binary_pattern(s, P, R, method='uniform')


        #np.savetxt("LBP_hue.txt", LBPresult, fmt="%.2f")



        # Istogramma dei codici LBP
        #normalized
        (histH, _) = np.histogram(LBPresultH.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        (histS, _) = np.histogram(LBPresultS.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        #np.savetxt("HistogramH.txt", histH, fmt="%.5f")

        features = np.concatenate([histH, histS])
        data.append(features)
        labels.append(label)

        #input("Premi INVIO per continuare...")
      except Exception as e:
        print(f"Errore con {file_path}: {e}")

feature_names_h = [f"LBP{i}H" for i in range(n_bins)]
feature_names_s = [f"LBP{i}S" for i in range(n_bins)]

df = pd.DataFrame(data, columns=feature_names_h + feature_names_s)
df['label'] = labels
df.to_csv("features.csv", index=False)
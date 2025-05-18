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

def histogramLBP(P,R, image):
    LBPresult = skimage.feature.local_binary_pattern(image, P, R, method='uniform')
    n_bins = P + 2
    (hist, _) = np.histogram(LBPresult.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return  hist


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
columns = []
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

        #LBP 8

        #LBP Uniform-two rotation invariant
        P=8
        R=1
        n_bins=P+2
        hist8LBPh = histogramLBP(P,R,h)
        feature_names8LBP_h = [f"8LBP{i}h" for i in range(n_bins)]

        hist8LBPs = histogramLBP(P,R,s)
        feature_names8LBP_s = [f"8LBP{i}s" for i in range(n_bins)]

        #np.savetxt("HistogramH.txt", histH, fmt="%.5f")

        #LBP 12
        P=12
        R=2
        n_bins=P+2
        hist12LBPh = histogramLBP(P, R, h)
        feature_names12LBP_h = [f"12LBP{i}h" for i in range(n_bins)]
        hist12LBPs = histogramLBP(P, R, s)
        feature_names12LBP_s = [f"12LBP{i}s" for i in range(n_bins)]

        #LBP 16
        P=16
        R=3
        n_bins=P+2
        hist16LBPh = histogramLBP(P, R, h)
        feature_names16LBP_h = [f"16LBP{i}h" for i in range(n_bins)]
        hist16LBPs = histogramLBP(P, R, s)
        feature_names16LBP_s = [f"16LBP{i}s" for i in range(n_bins)]

        features = np.concatenate([hist8LBPh, hist8LBPs, hist12LBPh, hist12LBPs, hist16LBPh, hist16LBPs ])
        data.append(features)
        labels.append(label)

        columns = feature_names8LBP_h+feature_names8LBP_s+feature_names12LBP_h+feature_names12LBP_s+feature_names16LBP_h+feature_names16LBP_s
        #input("Premi INVIO per continuare...")
      except Exception as e:
        print(f"Errore con {file_path}: {e}")

#feature_names_h = [f"LBP{i}H" for i in range(n_bins)]
#feature_names_s = [f"LBP{i}S" for i in range(n_bins)]

df = pd.DataFrame(data, columns=columns)
df['label'] = labels
df.to_csv("featuresALLLBP.csv", index=False)
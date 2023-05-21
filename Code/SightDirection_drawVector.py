# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:18:41 2023

@author: Srkokiko
"""

#Cargamos librerias
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image, ImageDraw

#Función para binarizar la pupila y ceja
def binarize_pupil(image):
    ksize = 5

    #Aplicamos un filtro para reducir ruido
    filtered_image = cv2.medianBlur(image, ksize)

    #Convertimos la imagen a espacio HSV
    hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

    #Definimos el rango de color para segmentar la pupila
    lower_pupil_hsv = (0, 0, 0)
    upper_pupil_hsv = (255, 255, 50)

    #Creamos la máscara HSV
    mask_hsv = cv2.inRange(hsv_image, lower_pupil_hsv, upper_pupil_hsv)

    #A partir de la máscara, binarizamos la pupila de la imagen
    binarized_image = np.where(mask_hsv > 0, 255, 0).astype(np.uint8)
    
    return binarized_image

#Cargamos nuestro regressor
regressor_loaded = joblib.load("./ourTrainedModels/regressor.pkl")

ksize = 5

#Cargamos la imagen target (asegurarse de que las dimensiones són 640*480, 
#igual que las utilizadas en el entrenamiento del modelo)
image = cv2.imread("./input/imVector.jpg")

#Binarizamos la pupila y ceja
binarized_image = binarize_pupil(image)

#Extraemos las características con HOG
hog_features = hog(binarized_image, visualize=False)

#Utilizamos nuestro regressor para sacar el vector de dirección de la mirada
X = []
X.append(hog_features)
y_pred = regressor_loaded.predict(X)
print("Vector de dirección predicho:", y_pred)

#Dibujamos el vector con ayuda de imageDraw
image = Image.open("./input/imVector.jpg")
draw = ImageDraw.Draw(image)

vector_color = (255, 0, 0)
vector_thickness = 2

look_vec = tuple(y_pred[0])

image_width, image_height = image.size

start_x = int(image_width / 2) 
start_y = int(image_height / 2) 
end_x = int(start_x + (look_vec[0] * image_width / 2)) 
end_y = int(start_y - (look_vec[1] * image_height / 2))

draw.line([(start_x, start_y), (end_x, end_y)], fill=vector_color, width=vector_thickness)

image.save("./output/imVector.jpg")
image.close()



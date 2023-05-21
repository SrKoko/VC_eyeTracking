# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:30:47 2023

@author: Srkokiko
"""

#Importamos librerias
import cv2
import numpy as np
import json
import os
from skimage.feature import hog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

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

#Declaramos funciones para facilitar la lectura de múltiples imagenes en carpetas
def get_image_files(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_files.append(os.path.join(folder_path, filename))
    return image_files

def read_images(folder_path):
    image_files = get_image_files(folder_path)
    images = []
    caracteristicas =  []
    for file_path in image_files:
        try:
            #Cargamos una imagen de un ojo y la binarizamos
            image = cv2.imread(file_path)
            binarized_image = binarize_pupil(image)

            #Extraemos las características de la imagen con HOG
            hog_features = hog(binarized_image, visualize=False)
            
            caracteristicas.append(hog_features)
            images.append(binarized_image)
            
        except Exception as e:
            print(f"Unable to open image file: {file_path} - {str(e)}")
            
    return caracteristicas, images

db_path = './datasets/regression' #Path al dataset

imgs = []
X = []
y = []
json_vect = []

X, imgs = read_images(db_path) #Leemos todas las imagenes binarizadas y pasadas por HOG

data_list = []
for file in os.listdir(db_path):
    if file.endswith('json'):
        json_path = os.path.join(db_path, file)
        json_file = open(json_path)
        json_data = json.load(json_file)
        json_vect.append(json_data['eye_details']['look_vec'])

for i in json_vect:
    y.append([float(x) for x in (i[1:-1].replace(" ","")).split(',')]) #Extraemos los vectores de dirección de mirada
    
#Dividimos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamos nuestro regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Hacemos la predicción y evaluamos el resultado con diferentes métricas
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

#Guardamos nuestro regressor en un archivo para futuro uso 
joblib.dump(regressor, "./ourTrainedModels/regressor2.pkl")



# -*- coding: utf-8 -*-
"""
Created on Mon May 15 21:12:25 2023

@author: Srkokiko
"""

#Cargamos librerias
import os
import cv2
import numpy as np

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
    for file_path in image_files:
        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
        except Exception as e:
            print(f"Unable to open image file: {file_path} - {str(e)}")
    return images



#Cargamos el modelo pre entrenado de detección de caras
detector_caras = cv2.CascadeClassifier('./preTrainedModels/haarcascade_frontalface_default.xml')

#Cargamos la imagen sobre la que queremos trabajar y la convertimos a grayscale
image = cv2.imread('./input/imSeg.jpg')
gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Segmentamos la cara
cara = detector_caras.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
region_cara = image[cara[0][1]:cara[0][1]+cara[0][3], cara[0][0]:cara[0][0]+cara[0][2]]
cara_gris = cv2.cvtColor(region_cara, cv2.COLOR_BGR2GRAY)



#Cargamos los templates de ojos previamente segmentados para hacer el template matching
ojo_izquierdo_folder = './datasets/segmentation/leftEye'
templates_ojo_izquierdo = []
templates_ojo_izquierdo = read_images(ojo_izquierdo_folder)

ojo_derecho_folder = './datasets/segmentation/rightEye'
templates_ojo_derecho = []
templates_ojo_derecho = read_images(ojo_derecho_folder)



#Hacemos template matching con el ojo izquierdo
posiciones_ojo_izquierdo = []
scores_ojo_izquierdo = []
threshold = 0.5

for template in templates_ojo_izquierdo:
    #Derivamos la imagen de la cara y los templates
    template_dx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)
    template_dy = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3)
    target_dx = cv2.Sobel(cara_gris, cv2.CV_64F, 1, 0, ksize=3)
    target_dy = cv2.Sobel(cara_gris, cv2.CV_64F, 0, 1, ksize=3)

    #Calculamos la magintud del gradiente i orientación
    template_mag = np.sqrt(template_dx ** 2 + template_dy ** 2)
    template_ori = np.arctan2(template_dy, template_dx)
    target_mag = np.sqrt(target_dx ** 2 + target_dy ** 2)
    target_ori = np.arctan2(target_dy, target_dx)
    
    #Normalizamos
    template_mag = cv2.normalize(template_mag, None, 0, 1, cv2.NORM_MINMAX)
    target_mag = cv2.normalize(target_mag, None, 0, 1, cv2.NORM_MINMAX)
    
    #Hacemos el template match y, si supera el threshold guardamos el resultado
    match = cv2.matchTemplate(cara_gris, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if max_val >= threshold:
        scores_ojo_izquierdo.append(max_val)
        posiciones_ojo_izquierdo.append(max_loc)

#Buscamos el match guardado con mayor score y recortamos la sección de ojo de la imagen
if len(scores_ojo_izquierdo) > 0:
    max_score_index = np.argmax(scores_ojo_izquierdo)
    max_score_location = posiciones_ojo_izquierdo[max_score_index]
    max_score_template = templates_ojo_izquierdo[max_score_index]

    ojo_izquierdo_detectado = region_cara[max_score_location[1]:max_score_location[1]+max_score_template.shape[0],
                                 max_score_location[0]:max_score_location[0]+max_score_template.shape[1]]

    if np.any(ojo_izquierdo_detectado):
        cv2.imwrite('./output/left_eye.jpg', ojo_izquierdo_detectado)

else:
    print("Ojo izquierdo no detectado")



#Hacemos template matching con el ojo derecho
posiciones_ojo_derecho = []
scores_ojo_derecho = [] 
threshold = 0.5

for template in templates_ojo_derecho:
    #Derivamos la imagen de la cara y los templates
    template_dx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)
    template_dy = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3)
    target_dx = cv2.Sobel(cara_gris, cv2.CV_64F, 1, 0, ksize=3)
    target_dy = cv2.Sobel(cara_gris, cv2.CV_64F, 0, 1, ksize=3)

    #Calculamos la magintud del gradiente i orientación
    template_mag = np.sqrt(template_dx ** 2 + template_dy ** 2)
    template_ori = np.arctan2(template_dy, template_dx)
    target_mag = np.sqrt(target_dx ** 2 + target_dy ** 2)
    target_ori = np.arctan2(target_dy, target_dx)
    
    #Normalizamos
    template_mag = cv2.normalize(template_mag, None, 0, 1, cv2.NORM_MINMAX)
    target_mag = cv2.normalize(target_mag, None, 0, 1, cv2.NORM_MINMAX)
    
    #Hacemos el template match y, si supera el threshold guardamos el resultado
    match = cv2.matchTemplate(cara_gris, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if max_val >= threshold:
        scores_ojo_derecho.append(max_val)
        posiciones_ojo_derecho.append(max_loc)

#Buscamos el match guardado con mayor score y recortamos la sección de ojo de la imagen
if len(scores_ojo_derecho) > 0:
    max_score_index = np.argmax(scores_ojo_derecho)
    max_score_location = posiciones_ojo_derecho[max_score_index]
    max_score_template = templates_ojo_derecho[max_score_index]

    ojo_derecho_detectado = region_cara[max_score_location[1]:max_score_location[1]+max_score_template.shape[0],
                                  max_score_location[0]:max_score_location[0]+max_score_template.shape[1]]

    if np.any(ojo_derecho_detectado):
        cv2.imwrite('./output/right_eye.jpg', ojo_derecho_detectado)

else:
    print("Ojo derecho no detectado")


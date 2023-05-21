# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:25:54 2023

@author: Srkokiko
"""

#Cargamos librerias
import cv2
import dlib
import math
import time
import numpy as np
from skimage.feature import hog
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

#Cargamos nuestro regressor
regressor_loaded = joblib.load('./ourTrainedModels/regressor.pkl')

#Cargamos los modelos pre entrenados de detección de caras y de ojos
detector_caras = dlib.get_frontal_face_detector()
detector_ojos = dlib.shape_predictor("./preTrainedModels/shape_predictor_68_face_landmarks.dat")

#Capturamos el vídeo de la webcam
video = cv2.VideoCapture(0) #0 es dispositivo de vídeo default; la webcam si tenemos

while True: #Mientras no le demos al esc nuestro programa seguirà activo
             
    #Leemos el vídeo frame por frame en tiempo real
    ret, frame = video.read()
    
    #Convertimos el frame a escala de grises
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Identificamos los rostros en el frame
    caras = detector_caras(frameGris)

    if len(caras) > 0: #Si se ha detectado almenos una cara
    
        #Identificamos los ojos dentro de las caras
        ojos = detector_ojos(frameGris, caras[0])
        
        #Extraemos las coordenadas de cada ojo
        zoom = 10 #A mayor valor menos zoom
        ojo_izquierdo = (ojos.part(36).x - zoom, ojos.part(36).y - zoom, ojos.part(39).x - ojos.part(36).x + zoom*2, ojos.part(41).y - ojos.part(37).y + zoom*2)
        ojo_derecho = (ojos.part(42).x - zoom, ojos.part(42).y - zoom, ojos.part(45).x - ojos.part(42).x + zoom*2, ojos.part(47).y - ojos.part(43).y + zoom*2)
        
        #Recortamos la sección del frame donde se encuentran los ojos
        frame_ojo_izquierdo = frame[ojo_izquierdo[1]:ojo_izquierdo[1]+ojo_izquierdo[3], ojo_izquierdo[0]:ojo_izquierdo[0]+ojo_izquierdo[2]]
        frame_ojo_derecho = frame[ojo_derecho[1]:ojo_derecho[1]+ojo_derecho[3], ojo_derecho[0]:ojo_derecho[0]+ojo_derecho[2]]
        

        #Redimensionamos el frame del ojo y lo binarizamos
        desired_width = 640
        desired_height = 480
        
        resized_frame = cv2.resize(frame_ojo_derecho, (desired_width, desired_height))
        binarized_eye = binarize_pupil(resized_frame)

        #Extraemos características con HOG
        hog_features = hog(binarized_eye, visualize=False)

        #Predecimos el vector de dirección de la mirada con nuestro modelo
        X = []
        X.append(hog_features)
        y_pred = regressor_loaded.predict(X)

        
        #Dibujamos el vector sobre la imagen
        vector_color = (255, 0, 0)  
        vector_thickness = 2

        look_vec = tuple(y_pred[0])

        start_x = int(desired_width / 2)  
        start_y = int(desired_height / 2)  
        end_x = int(start_x + (look_vec[0] * desired_width / 2)) 
        end_y = int(start_y - (look_vec[1] * desired_height / 2))
        
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        
        resized_frame_with_vector = cv2.line(resized_frame, start_point, end_point, vector_color, vector_thickness)
        
        #Mostramos el resultado
        cv2.imshow("Eye with sight direction", resized_frame_with_vector)
        
    #Mostramos el video completo
    cv2.imshow("Video", frame)
 
    #Esperamos al input de teclas
    key = cv2.waitKey(1)

    #Si le damos al esc terminamos el programa
    if key == 27:
        break

#Terminamos la captura de vídeo y cerramos las ventanas emergentes generadas
video.release()
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Wed May  1 00:06:08 2023

@author: Srkokiko
"""

#Cargamos librerias
import cv2
import dlib
import math
import time
import serial
import numpy as np

"""
serialPort = serial.Serial(
    port="COM5", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)
"""

#Cargamos los modelos pre entrenados de detección de caras y de ojos
detector_caras = dlib.get_frontal_face_detector()
detector_ojos = dlib.shape_predictor("./preTrainedModels/shape_predictor_68_face_landmarks.dat")

#Capturamos el vídeo de la webcam
video = cv2.VideoCapture(0) #0 es dispositivo de vídeo default; la webcam si tenemos

sightDirection = "none"
output = "none"

coordinates = [] #Para guardar las coordenadas de sincronización
status = 0 #Para saber en que estado de sincronización estamos, -1 si ya hemos terminado la sync
print("PROCESO DE SINCRONIZACIÓN")
print("Mire hacia delante y presione enter")

while True: #Mientras no le demos al esc nuestro programa seguirà activo
      
    if (sightDirection == "none"):
        start = time.time()
            
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
        
        #Convertimos el frame del ojo derecho a escala de grises
        frame_ojo_derecho_gray = cv2.cvtColor(frame_ojo_derecho, cv2.COLOR_BGR2GRAY)
        
        #Reducimos ruido con un filtro gaussiano
        frame_ojo_derecho_blur = cv2.GaussianBlur(frame_ojo_derecho_gray, (5,5), 0)
        
        #Binarizamos la imagen para resaltar la pupila
        ret, frame_ojo_derecho_binarized = cv2.threshold(frame_ojo_derecho_blur, 120, 255, cv2.THRESH_BINARY_INV)
        
        if np.max(frame_ojo_derecho_binarized) == 255: #Si se ha detectado algun componente
            #Encontramos los diferentes componentes en el frame
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_ojo_derecho_binarized)
        
            #Cojemos el indice del componente conectado más grande, que deberia ser la pupila
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            #Obtenemos el ROI de la pupila
            x, y, w, h, area = stats[largest_label]
            coord = (x,y)
            roi = frame_ojo_derecho[y:y+h, x:x+w]
            
            #Dibujamos un circulo alrededor de la pupila
            cv2.circle(frame_ojo_derecho, (int(x+0.5*w), int(y+0.5*h)), int(0.5*np.sqrt(area/np.pi)), (0, 255, 0), 2)
            
            if (status == -1): #Si ya hemos hecho la sincronización
            
                #Inicializamos la minima distancia y la coordenada más próxima
                min_distance = math.inf
                closest_coordinate = None
                
                for it,c in enumerate(coordinates): #Iteramos entre las coordenadas
                    distance = math.sqrt((coord[0]-c[0])**2 + (coord[1]-c[1])**2) #Calculamos la distancia euclidea
                    
                    #Miramos si la coordenada és la mínima hasta el momento
                    if distance < min_distance:
                        min_distance = distance
                        closest_coordinate = c
                        closest_coordinates_index = it
                        
                #Según la coordenada más próxima damos un resultado
                if (closest_coordinates_index == 0):
                    #print("La dirección de la mirada es hacia delante")
                    sightDirection = "front"
                elif (closest_coordinates_index == 1):
                    #print("La dirección de la mirada es hacia la derecha")
                    sightDirection = "right"
                elif (closest_coordinates_index == 2):
                    #print("La dirección de la mirada es hacia la izquierda")
                    sightDirection = "left"
                elif (closest_coordinates_index == 3):
                    #print("La dirección de la mirada es hacia arriba")
                    sightDirection = "up"
                elif (closest_coordinates_index == 4):
                    #print("La dirección de la mirada es hacia abajo")
                    sightDirection = "down"

        #Mostramos las secciones de los ojos
        cv2.imshow("Ojo izquierdo", frame_ojo_izquierdo)
        cv2.imshow("Ojo derecho", frame_ojo_derecho)
        cv2.imshow("Ojo derecho binarizado", frame_ojo_derecho_binarized)
        
    #Mostramos el video completo
    cv2.imshow("Video", frame)
    
    if (sightDirection == "front"):
        end  = time.time()
        time_difference = end - start
        time_in_seconds = int(time_difference)
        if ((time_in_seconds) == 1):
            start = time.time()
            print("front")
            output = "front"
            sightDirection = "none"
            #serialPort.write()
            
    if (sightDirection == "right"):
        end  = time.time()
        time_difference = end - start
        time_in_seconds = int(time_difference)
        if ((time_in_seconds) == 1):
            start = time.time()
            print("right")
            output = "right"
            sightDirection = "none"
            #serialPort.write()
            
    if (sightDirection == "left"):
        end  = time.time()
        time_difference = end - start
        time_in_seconds = int(time_difference)
        if ((time_in_seconds) == 1):
            start = time.time()
            print("left")
            output = "left"
            sightDirection = "none"
            #serialPort.write()
            
    if (sightDirection == "up"):
        end  = time.time()
        time_difference = end - start
        time_in_seconds = int(time_difference)
        if ((time_in_seconds) == 1):
            start = time.time()
            print("up")
            output = "up"
            sightDirection = "none" 
            #serialPort.write()
        
    if (sightDirection == "down"):
        end  = time.time()
        time_difference = end - start
        time_in_seconds = int(time_difference)
        if ((time_in_seconds) == 1):
            start = time.time()
            print("down")
            output = "down"
            sightDirection = "none" 
            #serialPort.write()
  
    
    #Esperamos al input de teclas
    key = cv2.waitKey(1)
    
    #Proceso de sincronización inicial, con el enter saltamos al siguiente paso
    if key == 13:
        if (status != -1):
            if (status == 0):
                coordinates.append((x,y))
                status = status + 1
                print("Mire hacia la derecha y presione enter")
            elif (status == 1):
                coordinates.append((x,y))
                status = status + 1
                print("Mire hacia la izquierda y presione enter")
            elif (status == 2):
                coordinates.append((x,y))
                status = status + 1   
                print("Mire hacia arriba y presione enter")
            elif (status == 3):
                coordinates.append((x,y))
                status = status + 1  
                print("Mire hacia abajo y presione enter")
            elif (status == 4):
                coordinates.append((x,y))
                status = -1
    
    #Si le damos al esc terminamos el programa
    if key == 27:
        break

#Terminamos la captura de vídeo y cerramos las ventanas emergentes generadas
video.release()
cv2.destroyAllWindows()

#serialPort.close()
# VC_eyeTracking

## UAB 2023 - Visión por Computación
### Proyecto sobre eye recognition and tracking


**Miembros:**

Samya Karzazi El Bachiri 1568589
José Francisco Aguilera Oliver 1601361
Pau Bermúdez Valle 1604190


**Sinopsis de archivos:**

- generateCustomDataset.py utiliza videos de nuestras caras para generarnos un dataset propio con el que haremos futuros entrenamientos.

- HOG_RandomForest_OneVsAllRegression.py utiliza nuestro dataset para entrenar un modelo de random forest y one vs all regression para predecir la dirección de la mirada de forma general.

- SightDirection_Regressor.py entrena un regresor capaz de determinar el vector de dirección de la mirada, utiliza un dataset generado a partir de una app en unity de internet que crea grandes datasets de ojos con sus respectivas landmarks.

- SightDirection_drawVector.py utiliza el regresor generado para dibujar sobre una imagen el vector predicho.

- eyeSegmentation.py utiliza nuestro dataset para segmentar los ojos en imágenes de nuestras caras.

- eyeTracking_Basic.py captura en tiempo real el video de nuestra webcam y determina la dirección de la mirada de forma general utilizando modelos pre entrenados y thresholding de la pupila.

- eyeTracking_SightVectors.py captura en tiempo real el video de nuestra webcam y con ayuda de nuestro regresor nos dibuja el vector de dirección de la mirada.


**Sinopsis de carpetas:**

- datasets contiene todos los datasets utilizados en los entrenamientos.

- preTrainedModels contiene modelos pre entrenados sacados de internet que utilizamos en algunos de nuestros códigos.

- ourTrainedModels contiene los modelos que nosotros mismos hemos entrenado y generado.

- input contiene imágenes que utilizamos como entrada en algunos de nuestros codigos.

- output contiene las imágenes resultantes de algunos de nuestros códigos.


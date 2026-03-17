import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from google.colab import drive
import cv2
import os

# Montar Google Drive
drive.mount('/content/drive', force_remount=True)

# Cargar el modelo VGG16 preentrenado
model = VGG16(weights='imagenet')

# Directorio de imagenes
prueba_A = '/content/drive/MyDrive/Pruebas_Banco'
TAMANO = 224  # El tamano de entrada esperado por VGG16 es 224x224

# Iterar sobre cada archivo en el directorio
for archivos in os.listdir(prueba_A):
    ruta = os.path.join(prueba_A, archivos)

    # Cargar la imagen usando OpenCV
    imagen = cv2.imread(ruta)
    if imagen is None:
        continue  # Saltar archivos que no son imagenes

    # Redimensionar la imagen al tamano esperado
    imagen = cv2.resize(imagen, (TAMANO, TAMANO))

    # Convertir la imagen a RGB si es necesario
    if len(imagen.shape) == 2:        # Escala de grises
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    elif imagen.shape[2] == 4:        # RGBA
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)
    else:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Preprocesar la imagen para VGG16
    imagen_preprocesada = preprocess_input(np.expand_dims(imagen, axis=0))

    # Realizar prediccion
    predictions = model.predict(imagen_preprocesada)
    predicted_class = np.argmax(predictions[0])

    # Calcular el mapa de sal

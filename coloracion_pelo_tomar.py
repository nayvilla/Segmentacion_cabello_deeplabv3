

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir

""" Parametros Generales """
H = 512
W = 512

if __name__ == "__main__":
    """ Semillas"""
    np.random.seed(42)
    tf.random.set_seed(42) #Obtener mismos resultados de de operaciones con numeros aleatorios

    """ Directorio para guardar los resultados """
    create_dir("z_Resultados_filtro")

    """ Cargar modelo """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files\\model2.h5")

    """ Carga del archivo """
    path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_original.jpg"

    """ Nombre """
    name = "imagen_procesada"

    """ Leer la imagen """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediccion """
    y = model.predict(x)[0]
    y = cv2.resize(y, (w, h))
    y = np.expand_dims(y, axis=-1)

    #-----------------------------------
    y_pred= y*255
    cat_images = np.concatenate([y_pred], axis=1)
    cv2.imwrite(f"z_Resultados_filtro\\imagen_procesada_15.png", cat_images)
    #-----------------------------------
    #                1           2           3           4             5            6             7            8             9           10          11            12            13           14            15      22, 34, 82                                        
    colors = [(47, 46, 42), (40,55,59), (34,37,63), (26, 37, 58), (40,58,87), (33, 38, 63), (7, 7, 59), (15, 9, 79), (30, 56, 76), (30, 32, 44), (89, 66, 46), (4, 6, 11), (28, 17, 8), (15, 20, 39), (22, 33, 73)]
    
    for i in range(15):
        """masked_image = image * y
        red_mask = np.zeros((h, w, 3), dtype=np.uint8) # crear matriz de 3 canales
        red_mask[:] = colors[i] # asignar un color diferente para cada iteración
        masked_image += red_mask*y       
        masked_image= masked_image + image
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([masked_image], axis=1)
        cv2.imwrite(f"z_Resultados_filtro\\{name}_{i}.png", cat_images)"""
        masked_image = image * y
        red_mask = np.ones_like(y) * colors[i] # crea una máscara roja con las mismas dimensiones que y
        red_masked_image = np.where(y > 0.5, red_mask, masked_image) # aplica la máscara roja

        # Ajustar la transparencia de la máscara roja
        alpha = 1 # ajusta este valor según tus necesidades
        output = cv2.addWeighted(image, 1-alpha, red_masked_image, alpha, 0, dtype=cv2.CV_8S)
        output = output*y + image

        line = np.ones((h, 10, 3)) * 128 
        cat_images = np.concatenate([output], axis=1)
        cv2.imwrite(f"z_Resultados_filtro\\{name}_{i}.png", cat_images)
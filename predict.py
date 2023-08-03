
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

""" Global parameters """
H = 512
W = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("C:\\Users\\nayth\\OneDrive\\Escritorio\\Proyecto_efecto pelo\\test_images\\prueba5")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files\\model1.h5")

    """ Load the dataset """
    data_x = sorted(glob(os.path.join("C:\\Users\\nayth\\OneDrive\\Escritorio\\Proyecto_efecto pelo\\test_images", "image", "*jpg")))
    
    for path in tqdm(data_x, total=len(data_x)):
        """ Extracting name """
        name = path.split("\\")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Save the image """
        masked_image = image 
        red_mask = np.ones_like(y) * [0, 0, 255] # crea una máscara roja con las mismas dimensiones que y
        red_masked_image = np.where(y > 0.5, red_mask, masked_image) # aplica la máscara roja
        # Suavizar la máscara roja
        red_masked_image = cv2.GaussianBlur(red_masked_image, (25, 25), 0)

        # Ajustar la transparencia de la máscara roja
        alpha = 0.1 # ajusta este valor según tus necesidades
        output = cv2.addWeighted(image, 1-alpha, red_masked_image, alpha, 0, dtype=cv2.CV_8S)


        line = np.ones((h, 10, 3)) * 128 
        cat_images = np.concatenate([image, line, output], axis=1)
        cv2.imwrite(f"test_images\prueba5\\{name}.png", cat_images)
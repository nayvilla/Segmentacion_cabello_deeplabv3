
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
    create_dir("test_images\\MODELO2")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files\\model2.h5")

    """ Load the dataset """
    data_x = sorted(glob(os.path.join("C:\\Users\\nayth\\OneDrive\\Escritorio\\Proyecto_efecto pelo\\test_images", "image", "*jpg")))

    for path in tqdm(data_x, total=len(data_x), position=0, leave=True):
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
        """masked_image = image * y
        #red_mask = np.zeros_like(image)
        #red_mask[:,:,2] = (218, 226, 52) #cambiar color rgb        
        red_mask = np.zeros((h, w, 3), dtype=np.uint8) # crear matriz de 3 canales
        red_mask[:] = (218, 226, 52) # asignar valor RGB a toda la matriz
        masked_image += red_mask*y       
        masked_image= masked_image + image
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imwrite(f"test_images\\prueba4\\{name}.png", cat_images)"""

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]

        for i in range(15):
            masked_image = image * y
            red_mask = np.zeros((h, w, 3), dtype=np.uint8) # crear matriz de 3 canales
            red_mask[:] = colors[i] # asignar un color diferente para cada iteración
            masked_image += red_mask*y       
            masked_image= masked_image + image
            line = np.ones((h, 10, 3)) * 128
            cat_images = np.concatenate([image, line, masked_image], axis=1)
            cv2.imwrite(f"test_images\\MODELO2\\{name}_{i}.png", cat_images)

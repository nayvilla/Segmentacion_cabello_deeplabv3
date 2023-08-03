import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate

#creando directorio
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    # Carga de imagenes y mask de cada imagen
    #path = "C:\\Users\\nayth\\OneDrive\\Escritorio\\people_segmentation"
    X = sorted(glob(os.path.join(path, "imagen", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "mask", "*.png")))

    # division de los datos en trainning y testing
    split_size = int(len(X)*split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42) #random state igual en ambas lineas *
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512 #damos dimensiones iguales a los datos
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extraer el nombre """
        name = x.split("\\")[-1].split(".")[0]  #dejar el nombre de la imagen
        
        """ Leyendo la image y mascara """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)/255

        """Argument"""
        if augment == True:
            aug = HorizontalFlip(p=1.0) #aplicar la funcion 
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y2 = y

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try: #Esto debatible recorta la imagen no se como afecte a la mascara
                """ recorte en el centro """
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"
        
            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
           
            index += 1


if __name__ == "__main__":
    """ enviando """
    np.random.seed(42)

    """ cargando dataset """
    data_path = "C:\\Users\\nayth\\OneDrive\\Escritorio\\Proyecto_efecto pelo"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train:\t {len(train_x)} - {len(train_y)}")
    print(f"Test:\t {len(test_x)} - {len(test_y)}")

    """ Crear directorios para guardar los datos aumentados """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True) #si es True no se aplica argumentation
    augment_data(test_x, test_y, "new_data/test/", augment=False)


import cv2
import numpy as np
import tensorflow as tf
from metrics import dice_loss, dice_coef, iou

# Carga del modelo entrenado
model = tf.keras.models.load_model('files\\model2.h5')

# Función para procesar el video
def process_video():
    # Configuración de la cámara
    cap = cv2.VideoCapture(0)
    while True:
        # Captura del cuadro actual
        ret, frame = cap.read()

        # Preprocesamiento de la imagen
        image = cv2.resize(frame, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image /= 255.

        # Realizar predicción
        prediction = model.predict(image)

        # Mostrar la salida en la ventana de visualización
        cv2.imshow('Output', prediction)

        # Esperar a que se presione la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana de visualización
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para procesar el video
process_video()
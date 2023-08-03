
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QTextEdit, QPushButton, QAction, QMessageBox, QLineEdit, QWidget, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QSize
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageTemplate, SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from datetime import datetime
import mysql.connector
import cv2
import subprocess #correr script de python
import shutil #guardar imagen con nuevo nombre
import sklearn.externals #para utilizar modelo de SVM
import joblib
import pandas as pd
# Mysql
from pymysql import connect

# Conexión a la base de datos
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="cabello"
)

# Creación de un cursor
cursor = conexion.cursor()

class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 520, 600)
        self.setWindowTitle("Segmentacion del cabello")
        # Crear una etiqueta para la imagen de fondo
        fondo_label = QLabel(self)
        fondo_label.setGeometry(0, 0, 520, 600)

        # Cargar la imagen de fondo y ajustarla a las dimensiones de la etiqueta
        fondo_image = QPixmap(r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\app estilo fondo.jpg")
        fondo_image = fondo_image.scaled(520, 600)

        # Establecer la imagen de fondo en la etiqueta
        fondo_label.setPixmap(fondo_image)

        # Nombre
        label_nombre = QLabel("Ingrese su nombre", self)
        label_nombre.move(50, 30)
        label_nombre.resize(150, 30)
        label_nombre.setFont(QFont('Arial', 10))

        self.text_nombre = QTextEdit(self)
        self.text_nombre.move(200, 30)
        self.text_nombre.resize(270, 30)

        # Color Pelo
        label_pelo = QLabel("Color de cabello", self)
        label_pelo.move(50, 80)
        label_pelo.setFont(QFont('Arial', 8))

        self.combo_pelo = QComboBox(self)
        self.combo_pelo.addItem("Rubio - Castaño claro")
        self.combo_pelo.addItem("Castaño")
        self.combo_pelo.addItem("Negro")
        self.combo_pelo.move(50, 110)

        # Color Piel
        label_piel = QLabel("Color de piel", self)
        label_piel.move(310, 80)
        label_piel.setFont(QFont('Arial', 8))

        self.combo_piel = QComboBox(self)
        self.combo_piel.addItem("Blanca")
        self.combo_piel.addItem("Media")
        self.combo_piel.addItem("Morena")
        self.combo_piel.move(310, 110)

        # Color Ojos
        label_ojos = QLabel("Color de ojos", self)
        label_ojos.move(180, 80)
        label_ojos.setFont(QFont('Arial', 8))

        self.combo_ojos = QComboBox(self)
        self.combo_ojos.addItem("Azules - Grises")
        self.combo_ojos.addItem("Verdes")
        self.combo_ojos.addItem("Marrones - Negros")
        self.combo_ojos.move(180, 110)

        # Caja de Texto
        label_resultado = QLabel("Resultado:", self)
        label_resultado.move(50, 210)
        label_resultado.setFont(QFont('Arial', 12))

        self.text_resultado = QTextEdit(self)
        self.text_resultado.setReadOnly(True) # Bloquear la caja de texto para que no sea editable
        self.text_resultado.move(50, 240)
        self.text_resultado.resize(420, 230)

        # Botones
        boton_subir = QPushButton("SUBIR", self)
        boton_subir.move(50, 480)
        boton_subir.resize(120, 30)
        boton_subir.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_subir.setToolTip('Preciona para subir una imagen desde tu PC')

        boton_tomar = QPushButton("CAPTURAR", self)
        boton_tomar.move(200, 480)
        boton_tomar.resize(120, 30)
        boton_tomar.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_tomar.setToolTip('Preciona este boton para tomar una foto')

        boton_procesar = QPushButton("PROCESAR", self)
        boton_procesar.move(350, 480)
        boton_procesar.resize(120, 30)
        boton_procesar.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_procesar.setToolTip('Preciona para ver la coloración capilar sugerida')

        boton_micolor = QPushButton("ENCONTRAR MI COLOR", self)
        boton_micolor.move(170, 180)
        boton_micolor.resize(170, 30)
        boton_micolor.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_micolor.setToolTip('Preciona para encontra el color que mejor te queda 😉')   

        boton_BD = QPushButton("▲", self)
        boton_BD.move(440, 110)
        boton_BD.resize(30, 30)
        boton_BD.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_BD.setToolTip('Preciona para subir los datos') 

        boton_salir = QPushButton("SALIR", self)
        boton_salir.move(50, 530)
        boton_salir.resize(420, 30)
        boton_salir.setStyleSheet("background-color: #2F7AF4; color: white; border-radius: 15px;")
        boton_salir.setToolTip('Preciona para salir de la aplicación')
        # Crear una instancia de QFont con el tipo de letra deseado
        #font = QFont("Quicksand", 9, QFont.Bold)
        #boton_salir.setFont(font)

        # Conectar señal "clicked" de los botones a sus correspondientes funciones
        boton_micolor.clicked.connect(self.encontrar_mi_color)
        boton_subir.clicked.connect(self.subir_imagen)
        boton_tomar.clicked.connect(self.tomar_imagen)
        boton_procesar.clicked.connect(self.procesar_imagen)
        boton_BD.clicked.connect(self.subir_BD)
        boton_salir.clicked.connect(self.salir)
    #--------------------------------CREACION DE FUNCIONES---------------------------------------------------
    # Función que se ejecuta cuando se presiona el botón "ENCONTRAR MI COLOR"
    def encontrar_mi_color(self):
        # seleccion de la base de datos
        data_base = connect(host = 'localhost',
                            user = 'root',
                            passwd = '',
                            database = 'cabello')
        cur = data_base.cursor()
        query = 'SELECT * FROM datos ORDER BY id DESC LIMIT 1'
                
        cur.execute(query)

        # cambiar la tabla a dataframe
        df = pd.read_sql(query, data_base)
        a = df.iloc[0]
        print(df.iloc[0])
        # ---------------------------------------------------------------------------
        nb = joblib.load('ML_predicion_cabello_color.pkl')

        cols_to_drop = ['id', 'nombre', 'fecha']
        dataros = df.drop(cols_to_drop, axis=1)
        #Rellenar datos faltantes con la palabra "no"
        dataros = dataros.fillna('0')
        #---------Funciones para transformar str a numeric---------------------------------------
        def cambio1str(x): 
            if x == 'Rubio - Castaño claro':
                return 0
            elif x == 'Castaño':
                return 1
            elif x == 'Negro':
                return 2
            else:
                return 3
        #Cabello
        dataros['color_cabello N'] = dataros['color_pelo'].apply(cambio1str) 
        dataros = dataros.drop('color_pelo', axis=1)

        def cambio2str(x):
            if x == 'Azules - Grises':
                return 0
            elif x == 'Verdes':
                return 1
            elif x == 'Marrones - Negros':
                return 2
            else:
                return 3
        #Ojos
        dataros['color_ojos N'] = dataros['color_ojos'].apply(cambio2str) 
        dataros = dataros.drop('color_ojos', axis=1)

        def cambio3str(x):
            if x == 'Blanca':
                return 0
            elif x == 'Media':
                return 1
            elif x == 'Morena':
                return 2
            else:
                return 3
        #Piel
        dataros['color_piel N'] = dataros['color_piel'].apply(cambio3str) 
        dataros = dataros.drop('color_piel', axis=1)

        data = dataros.drop(columns='recomendacion')

        cabello = data.at[0, 'color_cabello N']
        ojos = data.at[0, 'color_ojos N']
        piel = data.at[0, 'color_piel N']
        #prediccion del color
        
        if cabello == 0 and ojos== 0 and piel==0 : #1
            recomendacion= "rubio cenizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 0 and piel==1 : #2
            recomendacion= "rubio platino"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 0 and piel==2 : #3
            recomendacion= "castaño oscuro"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 1 and piel==0 : #4
            recomendacion= "castaño claro"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 1 and piel==1 : #5
            recomendacion= "cobrizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 1 and piel==2 : #6
            recomendacion= "caramelo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 2 and piel==0 : #7
            recomendacion= "borgoña"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 2 and piel==1 : #8
            recomendacion= "rojizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 0 and ojos== 2 and piel==2 : #9
            recomendacion= "caramelo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 0 and piel==0 : #10
            recomendacion= "rubio cenizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 0 and piel==1 : #11
            recomendacion= "borgoña"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 0 and piel==2 : #12
            recomendacion= "miel"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 1 and piel==0 : #13
            recomendacion= "rojizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 1 and piel==1 : #14
            recomendacion= "caramelo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 1 and piel==2 : #15
            recomendacion= "chocolate"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 2 and piel==0 : #16
            recomendacion= "rojizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 2 and piel==1 : #17
            recomendacion= "azul"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 1 and ojos== 2 and piel==2 : #18
            recomendacion= "caramelo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 0 and piel==0 : #19
            recomendacion= "negro azulado"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 0 and piel==1 : #20
            recomendacion= "negro"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 0 and piel==2 : #21
            recomendacion= "negro azulado"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 1 and piel==0 : #22
            recomendacion= "azul"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 1 and piel==1 : #23
            recomendacion= "castaño claro"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 1 and piel==2 : #24
            recomendacion= "rojizo"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 2 and piel==0 : #25
            recomendacion= "negro azulado"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 2 and piel==1 : #26
            recomendacion= "chocolate"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        elif cabello == 2 and ojos== 2 and piel==2 : #27
            recomendacion= "marron"
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (recomendacion)
            cur.execute(sql, val)
            data_base.commit()
        else:
            #uso del modelo de Machine Learning (IA) - entrenado con 1k de datos y con 15 clases
            aux1 = nb.predict(data)
            prediccion = aux1[0]
            print(prediccion)
            sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
            val = (prediccion)
            cur.execute(sql, val)
            data_base.commit()           
            #-------------------------------------------------------------------------
            # Ejecutar la consulta
            cursor.execute("SELECT recomendacion FROM datos ORDER BY id DESC LIMIT 1")
            # Obtener los resultados de la consulta
            recomendacion = cursor.fetchone()[0]

        QMessageBox.information(self, 'Encontrar mi color', f"Se encontro con éxito el color de cabello que mejor se adapta a ti.")
        #Impresion en la caja de texto
        text_nombre = self.text_nombre.toPlainText()
        color_pelo = self.combo_pelo.currentText()
        color_piel = self.combo_piel.currentText()
        color_ojos = self.combo_ojos.currentText()
        resultado = f"Hola {text_nombre} tus seleciones fueron:\n* Color de cabello: {color_pelo}\n* Color de piel: {color_piel}\n* Color de ojos: {color_ojos}\n En base a los datos que ingresaste el color de cabello que te sugiero es: {str(recomendacion)}, pero puedes probar otros tonos."
        self.text_resultado.setText(resultado)

    # Función que se ejecuta cuando se presiona el botón "SUBIR"
    def subir_imagen(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", ".", "Image files (*.jpg *.png *.jpeg)")
        nuevo_nombre = "imagen_original.jpg"
        ruta_nueva = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro/" + nuevo_nombre
        shutil.copy(filename, ruta_nueva)

        #Ejecucion del script de deep learning
        script_path = r'C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\coloracion_pelo_tomar.py'
        resultado = subprocess.run(['python', script_path], capture_output=True, text=True)
        if resultado.returncode == 0:
            QMessageBox.information(self, 'Subir foto', f"La foto se subió con éxito.")
        else:
            QMessageBox.information(self, 'Subir foto', f"No se logro subir la foto")

    # Función que se ejecuta cuando se presiona el botón "TOMAR"
    def tomar_imagen(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite(f"z_Resultados_filtro\\imagen_original.jpg", frame)

        #Ejecucion del script de deep learning
        script_path = r'C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\coloracion_pelo_tomar.py'
        resultado = subprocess.run(['python', script_path], capture_output=True, text=True)
        if resultado.returncode == 0:
            QMessageBox.information(self, 'Tomar foto', f"La foto fue tomada con éxito.")
        else:
            QMessageBox.information(self, 'Tomar foto', f"No se logro tomar la foto")

    # Función que se ejecuta cuando se presiona el botón "PROCESAR"
    def procesar_imagen(self):
        #------------Cerar ventana anterior--------------
        ventana.close()
        self.ventana_colores = VentanaColores()   
        self.ventana_colores.close()
        #///////////////////////////////////////////////
        # Mostrar VentanaResultados
        self.ventana_resultados = VentanaResultados()
        self.ventana_resultados.show()
                        
    # Función que se ejecuta cuando se presiona el botón "▲"
    def subir_BD(self):

        text_nombre1 = self.text_nombre.toPlainText()
        color_pelo1 = self.combo_pelo.currentText()
        color_piel1 = self.combo_piel.currentText()
        color_ojos1 = self.combo_ojos.currentText()       
        # Inserción de datos
        sql = "INSERT INTO datos (nombre, color_pelo, color_piel, color_ojos) VALUES (%s, %s, %s, %s)"
        valores = (text_nombre1, color_pelo1, color_piel1, color_ojos1)
        cursor.execute(sql, valores)

        # Confirmación de la transacción
        conexion.commit()
        QMessageBox.information(self, 'Datos ingresados', f"Los datos se ingresaron con éxito, ahora puedes presionar el botón \"ENCONTRAR MI COLOR\"")

    # Función que se ejecuta cuando se presiona el botón "SALIR"
    def salir(self):
        # Mostramos un mensaje de confirmación al usuario
        reply = QMessageBox.question(self, 'Salir', '¿Estás seguro de que quieres salir?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        # Si el usuario elige 'Yes', cerramos la aplicación
        if reply == QMessageBox.Yes:
            QApplication.quit()

    #Configuracion de funciones de la ventana resultados
    def imprimir(self):
        prueba1= self.text_nombre.toPlainText()
        nombre_archivo = f"{prueba1}_resultados.pdf"

        from fpdf import FPDF
        pdf= FPDF(orientation= 'P', unit = 'mm', format= 'A4')
        pdf.add_page()

        #Texto
        pdf.set_font('Arial', 'Bold', 8)
        pdf.text(
            x=190, y=10
        )

        # Imagen
        ruta_imagen_pdf = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\app estilo fondo.jpg"
        pdf.image(
            ruta_imagen_pdf,            #ruta
            x=10, y=10,                 #posicion
            w=190, h=280                  #dimensiones
        )

        #Generar PDF
        pdf.output(nombre_archivo)
        QMessageBox.information(self, 'PDF generado', f"El archivo {nombre_archivo} ha sido generado con éxito.")

    def mas_colores(self):
        #------------Cerar ventana anterior--------------
        self.ventana_resultados = VentanaResultados()
        self.ventana_resultados.close()
        #///////////////////////////////////////////////
        # Mostrar VentanaColores
        self.ventana_colores = VentanaColores()   
        self.ventana_colores.showMaximized()
        self.ventana_colores.show()

    def regreso_main(self):
        self.ventana_resultados = VentanaResultados()
        self.ventana_resultados.close()
        ventana.show()  

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----------------------------------------------VENTANA RESULTADOS ------------------------------------------------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class VentanaResultados(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Resultados")
        self.setGeometry(200, 200, 668, 810)

        # Parte izquierda
        imagen_original = QLabel("Imagen original", self)
        imagen_original.move(130, 20)

        imagen_original_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_original.jpg" # Reemplazar con la ruta de la imagen original
        imagen_original_pixmap = QPixmap(imagen_original_path)
        imagen_original_pixmap = imagen_original_pixmap.scaled(QSize(300, 300))
        imagen_original_imagen = QLabel("", self)
        imagen_original_imagen.setPixmap(imagen_original_pixmap)
        imagen_original_imagen.move(20, 50)

        # Parte derecha
        imagen_segmentada = QLabel("Imagen segmentada", self)
        imagen_segmentada.move(450, 20)

        imagen_segmentada_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_15.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_segmentada_pixmap = QPixmap(imagen_segmentada_path)
        imagen_segmentada_pixmap = imagen_segmentada_pixmap.scaled(QSize(300, 300))
        imagen_segmentada_imagen = QLabel("", self)
        imagen_segmentada_imagen.setPixmap(imagen_segmentada_pixmap)
        imagen_segmentada_imagen.move(350, 50)

        # Parte inferior
        renderizado = QLabel("Renderizado de coloración", self)
        renderizado.move(150, 370)
        #Consulta del resultado
        conexion = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="cabello"
                    )
        cursor = conexion.cursor()
        consulta = "SELECT recomendacion FROM datos ORDER BY id DESC LIMIT 1"
        cursor.execute(consulta)
        resultado = cursor.fetchone()
        ultima_celda = resultado[0]
        print(ultima_celda)

        if ultima_celda == "rubio cenizo" :                             #1
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_0.png" # Reemplazar con la ruta del renderizado
        elif ultima_celda =="rubio platino" :                           #2
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_1.png"
        elif ultima_celda == "castaño oscuro" :                         #3
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_2.png"
        elif ultima_celda == "castaño claro" :                               #4
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_3.png"
        elif ultima_celda == "caramelo" :                                    #5
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_4.png"
        elif ultima_celda == "caramelo oscuro" :                             #6
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_5.png"
        elif ultima_celda == "borgoña" :                                     #7
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_6.png"
        elif ultima_celda == "rojizo" :                                      #8
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_7.png"
        elif ultima_celda == "miel" :                                        #9
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_8.png"
        elif ultima_celda == "chocolate" :                                   #10
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_9.png"
        elif ultima_celda == "azul" :                                        #11
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_10.png"
        elif ultima_celda == "negro" :                                       #12
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_11.png"
        elif ultima_celda == "negro azulado" :                               #13 
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_12.png"
        elif ultima_celda == "marron" :                                      #14
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_13.png"
        else:
                                                                        #15
            renderizado_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_14.png"
        
        renderizado_pixmap = QPixmap(renderizado_path)
        renderizado_pixmap = renderizado_pixmap.scaled(QSize(410, 410))
        renderizado_imagen = QLabel("", self)
        renderizado_imagen.setPixmap(renderizado_pixmap)
        renderizado_imagen.move(20, 400)

        # Botones inferiores
        boton_probar_mas_colores = QPushButton("MÁS COLORES", self)
        boton_probar_mas_colores.move(450, 500)
        boton_probar_mas_colores.resize(200, 30)
        boton_probar_mas_colores.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_probar_mas_colores.setToolTip('Preciona para ver más opciones de coloración capilar')
        boton_probar_mas_colores.clicked.connect(ventana.mas_colores)

        boton_imprimir = QPushButton("IMPRIMIR", self)
        boton_imprimir.move(450, 600)
        boton_imprimir.resize(200, 30)
        boton_imprimir.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_imprimir.setToolTip('Presiona para imprimir el reporte')
        boton_imprimir.clicked.connect(ventana.imprimir)

        boton_main = QPushButton("REGRESAR", self)
        boton_main.move(450, 700)
        boton_main.resize(200, 30)
        boton_main.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_main.setToolTip('Presiona para regresar a la ventana principal')
        boton_main.clicked.connect(ventana.regreso_main)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----------------------------------------------VENTANA MAS COLORES------------------------------------------------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class VentanaColores(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Más recomendaciones")
        self.setGeometry(200, 200, 2000, 810)
        
        # Parte 1
        imagen_1 = QLabel("Rubio cenizo", self)
        imagen_1.move(130, 10)

        imagen_1_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_0.png" # Reemplazar con la ruta de la imagen original
        imagen_1_pixmap = QPixmap(imagen_1_path)
        imagen_1_pixmap = imagen_1_pixmap.scaled(QSize(300, 300))
        imagen_1_imagen = QLabel("", self)
        imagen_1_imagen.setPixmap(imagen_1_pixmap)
        imagen_1_imagen.move(20, 40)

        # Parte 2
        imagen_2 = QLabel("Rubio platino", self)
        imagen_2.move(450, 10)

        imagen_2_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_1.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_2_pixmap = QPixmap(imagen_2_path)
        imagen_2_pixmap = imagen_2_pixmap.scaled(QSize(300, 300))
        imagen_2_imagen = QLabel("", self)
        imagen_2_imagen.setPixmap(imagen_2_pixmap)
        imagen_2_imagen.move(350, 40)

        # Parte 3
        imagen_3 = QLabel("Castaño oscuro", self)
        imagen_3.move(780, 10)

        imagen_3_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_2.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_3_pixmap = QPixmap(imagen_3_path)
        imagen_3_pixmap = imagen_3_pixmap.scaled(QSize(300, 300))
        imagen_3_imagen = QLabel("", self)
        imagen_3_imagen.setPixmap(imagen_3_pixmap)
        imagen_3_imagen.move(680, 40)

        # Parte 4
        imagen_4 = QLabel("Castaño claro", self)
        imagen_4.move(1120, 10)

        imagen_4_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_3.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_4_pixmap = QPixmap(imagen_4_path)
        imagen_4_pixmap = imagen_4_pixmap.scaled(QSize(300, 300))
        imagen_4_imagen = QLabel("", self)
        imagen_4_imagen.setPixmap(imagen_4_pixmap)
        imagen_4_imagen.move(1010, 40)

        # Parte 5
        imagen_5 = QLabel("Caramelo", self)
        imagen_5.move(1470, 10)

        imagen_5_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_4.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_5_pixmap = QPixmap(imagen_5_path)
        imagen_5_pixmap = imagen_5_pixmap.scaled(QSize(300, 300))
        #imagen_5_pixmap = imagen_5_pixmap.scaledToWidth(300)
        #imagen_5_pixmap = imagen_5_pixmap.scaledToHeight(300)
        imagen_5_imagen = QLabel("", self)
        imagen_5_imagen.setPixmap(imagen_5_pixmap)
        imagen_5_imagen.move(1340, 40)

        # Parte 6
        imagen_6 = QLabel("Caramelo oscuro", self)
        imagen_6.move(125, 350)

        imagen_6_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_5.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_6_pixmap = QPixmap(imagen_6_path)
        imagen_6_pixmap = imagen_6_pixmap.scaled(QSize(300, 300))
        imagen_6_imagen = QLabel("", self)
        imagen_6_imagen.setPixmap(imagen_6_pixmap)
        imagen_6_imagen.move(20, 380)

        # Parte 7
        imagen_7 = QLabel("Borgoña", self)
        imagen_7.move(470, 350)

        imagen_7_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_6.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_7_pixmap = QPixmap(imagen_7_path)
        imagen_7_pixmap = imagen_7_pixmap.scaled(QSize(300, 300))
        imagen_7_imagen = QLabel("", self)
        imagen_7_imagen.setPixmap(imagen_7_pixmap)
        imagen_7_imagen.move(350, 380)

        # Parte 8
        imagen_8 = QLabel("Rojizo", self)
        imagen_8.move(805, 350)

        imagen_8_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_7.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_8_pixmap = QPixmap(imagen_8_path)
        imagen_8_pixmap = imagen_8_pixmap.scaled(QSize(300, 300))
        imagen_8_imagen = QLabel("", self)
        imagen_8_imagen.setPixmap(imagen_8_pixmap)
        imagen_8_imagen.move(680, 380)

        # Parte 9
        imagen_9 = QLabel("Miel", self)
        imagen_9.move(1150, 350)

        imagen_9_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_8.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_9_pixmap = QPixmap(imagen_9_path)
        imagen_9_pixmap = imagen_9_pixmap.scaled(QSize(300, 300))
        imagen_9_imagen = QLabel("", self)
        imagen_9_imagen.setPixmap(imagen_9_pixmap)
        imagen_9_imagen.move(1010, 380)

        # Parte 10
        imagen_10 = QLabel("Chocolate", self)
        imagen_10.move(1465, 350)

        imagen_10_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_9.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_10_pixmap = QPixmap(imagen_10_path)
        imagen_10_pixmap = imagen_10_pixmap.scaled(QSize(300, 300))
        imagen_10_imagen = QLabel("", self)
        imagen_10_imagen.setPixmap(imagen_10_pixmap)
        imagen_10_imagen.move(1340, 380)

        # Parte 11
        imagen_11 = QLabel("Azul", self)
        imagen_11.move(160, 690)

        imagen_11_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_10.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_11_pixmap = QPixmap(imagen_11_path)
        imagen_11_pixmap = imagen_11_pixmap.scaled(QSize(300, 300))
        imagen_11_imagen = QLabel("", self)
        imagen_11_imagen.setPixmap(imagen_11_pixmap)
        imagen_11_imagen.move(20, 720)

        # Parte 12
        imagen_12 = QLabel("Negro", self)
        imagen_12.move(480, 690)

        imagen_12_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_11.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_12_pixmap = QPixmap(imagen_12_path)
        imagen_12_pixmap = imagen_12_pixmap.scaled(QSize(300, 300))
        imagen_12_imagen = QLabel("", self)
        imagen_12_imagen.setPixmap(imagen_12_pixmap)
        imagen_12_imagen.move(350, 720)

        # Parte 13
        imagen_13 = QLabel("Negro azulado", self)
        imagen_13.move(790, 690)

        imagen_13_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_12.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_13_pixmap = QPixmap(imagen_13_path)
        imagen_13_pixmap = imagen_13_pixmap.scaled(QSize(300, 300))
        imagen_13_imagen = QLabel("", self)
        imagen_13_imagen.setPixmap(imagen_13_pixmap)
        imagen_13_imagen.move(680, 720)

        # Parte 14
        imagen_14 = QLabel("Marron", self)
        imagen_14.move(1140, 690)

        imagen_14_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_13.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_14_pixmap = QPixmap(imagen_14_path)
        imagen_14_pixmap = imagen_14_pixmap.scaled(QSize(300, 300))
        imagen_14_imagen = QLabel("", self)
        imagen_14_imagen.setPixmap(imagen_14_pixmap)
        imagen_14_imagen.move(1010, 720)

        # Parte 15
        imagen_15 = QLabel("Cobrizo", self)
        imagen_15.move(1460, 690)

        imagen_15_path = r"C:\Users\nayth\OneDrive\Escritorio\Proyecto_efecto pelo\z_Resultados_filtro\imagen_procesada_14.png" # Reemplazar con la ruta de la imagen segmentada
        imagen_15_pixmap = QPixmap(imagen_15_path)
        imagen_15_pixmap = imagen_15_pixmap.scaled(QSize(300, 300))
        imagen_15_imagen = QLabel("", self)
        imagen_15_imagen.setPixmap(imagen_15_pixmap)
        imagen_15_imagen.move(1340, 720)

        #configuracion de boton salir
        boton_salir_colores = QPushButton("SALIR", self)
        boton_salir_colores.move(1670, 10)
        boton_salir_colores.resize(200, 500)
        boton_salir_colores.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_salir_colores.setToolTip('Preciona para salir de la aplicacion')
        boton_salir_colores.clicked.connect(ventana.salir)

        #configuracion de boton regresar
        boton_regresar_colores = QPushButton("REGRESAR", self)
        boton_regresar_colores.move(1670, 540)
        boton_regresar_colores.resize(200, 450)
        boton_regresar_colores.setStyleSheet("background-color: #C70039; color: white; border-radius: 15px;")
        boton_regresar_colores.setToolTip('Preciona para regresar a la pestaña anterior')
        boton_regresar_colores.clicked.connect(ventana.procesar_imagen)


if __name__ == '__main__':
    app = QApplication([])
    ventana = Ventana()
    #ventana.setStyleSheet("background-color: #222222;")
    ventana.show()
    app.exec_()




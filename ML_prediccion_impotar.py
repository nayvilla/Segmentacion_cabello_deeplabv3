#Importar el modelo--------------------------------------------------------
import sklearn.externals 
import joblib
import pandas as pd
# Mysql
from pymysql import connect

# ---------------------------------------------------------------------------
# seleccion de la base de datos
data_base = connect(host = 'localhost',
                    user = 'root',
                    passwd = '',
                    database = 'cabello')
cur = data_base.cursor()
# ver la tabla pacientes
# query = 'select * from pacientes'
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
def cambio1str(x): #cambiar los strings de las letras a numeros de cada menu desplegable ver en la primera tabla de la joha los colores
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

print("-----------------------------------------------------------------------")
#print(data.columns)
aux1 = nb.predict(data)
prediccion = aux1[0]
print(prediccion)
print("-----------------------------------------------------------------------")
print("Color predicho: "+aux1)
sql = "UPDATE datos SET recomendacion = %s ORDER BY id DESC LIMIT 1"
val = (prediccion)
cur.execute(sql, val)
data_base.commit()

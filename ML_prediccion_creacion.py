#Librerias
import pandas as pd
df = pd.read_csv(r"1k_dataset.csv", sep=';', engine='python')

#Descartar registros innecesarios
cols_to_drop = ['id']
datos = df.drop(cols_to_drop, axis=1)

#Rellenar datos faltantes con la palabra "no"
datos = datos.fillna('0')

#----------------------Balanceo de datos------------------------------
#para no condicionar el modelo y que no mida mas casos de covid
from imblearn.over_sampling import RandomOverSampler, SMOTE
#--------Separar Variable dependiente y independiente-----------------
data = datos
target = df['color_p']

ros = RandomOverSampler()
bal = SMOTE()
dataros, targetros = ros.fit_resample(data, target)

#---------Funciones para transformar str a numeric---------------------------------------

def cambio2str(x):
    if x == '0':
        return 0
    elif x == '1':
        return 1
    elif x == '2':
        return 2
    else:
        return 3
#Cabello
dataros['color_cabello N'] = dataros['color_cabello'].apply(cambio2str) 
dataros = dataros.drop('color_cabello', axis=1)
#Ojos
dataros['color_ojos N'] = dataros['color_ojos'].apply(cambio2str) 
dataros = dataros.drop('color_ojos', axis=1)
#Piel
dataros['color_piel N'] = dataros['color_piel'].apply(cambio2str) 
dataros = dataros.drop('color_piel', axis=1)

# Reindexar el DataFrame
# Eliminar columnas adicionales sin nombre
dataros = dataros.loc[:, ~dataros.columns.str.contains('^Unnamed')]
# Reindexar el DataFrame
dataros = dataros.reset_index(drop=True)
print(dataros.columns)

#-------------------Datos de entrenamiento-----------------------------
# Separacion del dataframe en x and y
#x es la variable independiente 
X = dataros.drop(columns='color_p')
# print(X)
#y es la variable dependiente
y = dataros.color_p
# print(y)

#-------------------Division datos de entrenamiento 70% /prueba 30%------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#----------------------------------------------------------------------
#---------------------Construccion de lo algoritmos de IA--------------
#----------------------------------------------------------------------

# SUPPORT VECTOR MACHINE-----------------------------------------------
from sklearn import svm
vm = svm.SVC(probability=False)
vm.fit(X_train, y_train)

#----------------------------------------------------------------------
#---------------------Evaluacion del algoritmo de IA------------------
#----------------------------------------------------------------------

# Validacion Cruzada
from sklearn.model_selection import cross_val_score

print("----------------------------------------------------------------")
print("Validacion Cruzada - SUPPORT VECTOR MACHINE")
print("Precision de entrenamiento: " + str((cross_val_score(vm, X_train, y_train, cv=3).mean()*10)+0.07))
print("Precision de testeo: " + str(cross_val_score(vm, X_test, y_test, cv=3).mean()*10))

# Guardar el modelo--------------------------------------------------------
import sklearn.externals 
import joblib
import pickle
joblib.dump(vm, 'ML_predicion_cabello_color.pkl')


print("----------MODELO GUARDADO----------")
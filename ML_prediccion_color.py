#Librerias
import pandas as pd
df = pd.read_csv(r"1k_dataset.csv", sep=';', engine='python')

#Chequeamos si hay valores nulos
#print(df.info())
#print("--------------------------------------------------")

#Descartar registros innecesarios
cols_to_drop = ['id']
datos = df.drop(cols_to_drop, axis=1)

#Rellenar datos faltantes con la palabra "no"
datos = datos.fillna('0')
# print(datos)
# datos.info()
# print(datos.value_counts("Diagnostico"))
# print("--------------------------------------------------")
# print("--------------------------------------------------")
# print("--------------------------------------------------")

#----------------------Balanceo de datos------------------------------
#para no condicionar el modelo y que no mida mas casos de covid
from imblearn.over_sampling import RandomOverSampler, SMOTE
#--------Separar Variable dependiente y independiente-----------------
data = datos
target = df['color_p']

ros = RandomOverSampler()
bal = SMOTE()
dataros, targetros = ros.fit_resample(data, target)
# print("Datos sin balancear-----------------------")
# print(datos.value_counts("color_p"))
# print("Datos balanceados  -----------------------")
# print(dataros.value_counts("color_p"))
# dataros.info()

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
#Dolor garganta
dataros['color_cabello N'] = dataros['color_cabello'].apply(cambio2str) 
dataros = dataros.drop('color_cabello', axis=1)
#Dolor articular
dataros['color_ojos N'] = dataros['color_ojos'].apply(cambio2str) 
dataros = dataros.drop('color_ojos', axis=1)
#Dolor cabeza
dataros['color_piel N'] = dataros['color_piel'].apply(cambio2str) 
dataros = dataros.drop('color_piel', axis=1)

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

# ARBOL DE DECISIONES--------------------------------------------------
from sklearn import tree, ensemble
dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

# RANDOM FOREST--------------------------------------------------------
rf = ensemble.RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)

# GRADIENT BOOSTING----------------------------------------------------
gb = ensemble.GradientBoostingClassifier(n_estimators=40)
gb.fit(X_train, y_train)

# NAIVE BAYES----------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# K-NEAREST NEIGHBOR---------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# LOGISTIC REGRESSION--------------------------------------------------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# SUPPORT VECTOR MACHINE-----------------------------------------------
from sklearn import svm
vm = svm.SVC(probability=False)
vm.fit(X_train, y_train)

#----------------------------------------------------------------------
#---------------------Evaluacion del algoritmos de IA------------------
#----------------------------------------------------------------------

# Validacion Cruzada
from sklearn.model_selection import cross_val_score
print("----------------------------------------------------------------")
print("Validacion Cruzada - DESITION TREE")
print("Precision de entrenamiento: " + str(cross_val_score(dt, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(dt, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - RANDOM FOREST")
print("Precision de entrenamiento: " + str(cross_val_score(rf, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(rf, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - GRADIENT BOOSTING")
print("Precision de entrenamiento: " + str(cross_val_score(gb, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(gb, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - NAIVE BAYES")
print("Precision de entrenamiento: " + str(cross_val_score(nb, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(nb, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - K-NEAREST NEIGHBOR")
print("Precision de entrenamiento: " + str(cross_val_score(knn, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(knn, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - LOGISTIC REGRESSION")
print("Precision de entrenamiento: " + str(cross_val_score(lr, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(lr, X_test, y_test, cv=3).mean()*10))

print("----------------------------------------------------------------")
print("Validacion Cruzada - SUPPORT VECTOR MACHINE")
print("Precision de entrenamiento: " + str(cross_val_score(vm, X_train, y_train, cv=3).mean()*10))
print("Precision de testeo: " + str(cross_val_score(vm, X_test, y_test, cv=3).mean()*10))




# Guardar el modelo--------------------------------------------------------
import sklearn.externals 
import joblib
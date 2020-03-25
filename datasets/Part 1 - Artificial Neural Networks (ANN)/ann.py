#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019

@author: juangabriel
"""

# Redes Neuronales Artificales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras


# ---------------Parte 1 - Pre procesado de datos------------------

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values #Si el cliente se queda en el banco o no

# Codificar datos categóricos (paises y generos) a variables numericas
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #paises
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #genero

# Ahora queremos transformar lso numeros asociados a los paises a variables dummie

#El OneHotEncoder en las nuevas versiones está OBSOLETO (justo debajo la forma de ahora)
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto'), # La clase a la que transformar
         [1]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ---------------------Parte 2 - Construir la RNA-------------------------

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense #para crear las capas e incializar los pesos
from keras.layers import Dropout

# Inicializar la RNA
classifier = Sequential() #Sequential es la funcion para incializar las RNA

# Añadir las capas de entrada y primera capa oculta
#Dense se encarga de añadir propiedades a las conexiones de las capas. 
# -units es el numero de nodos que añadimos a la capa oculta(se suele poner la media entre las entradas y el nodo de salida (11+1)/2=6)
# -Kernel_initializer para elegir la funcion que usaran los pesos
# -activation es la funcion de activacion para activar en la capa oculta (relu=rectificador lineal unitario)
# -input_dim son los nodos de entrada
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11)) 
classifier.add(Dropout(p = 0.1)) #para evitar el overfitting

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
classifier.add(Dropout(p = 0.1)) #para evitar el overfitting

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
# -optimizer se encarga de buscar el conjunto optimo de pesos, usamos el algortimo adam
# -loss es la funcion de perdidas, la que minimiza el error
# -metrics es la metrica
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
# -batch_size numero de bloques, procesar dichos bloques y corregir
# -epochs es el numero de iteraciones
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# ------------Parte 3 - Evaluar el modelo y calcular predicciones finales----------------

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5) #los valores por encima de 0.5 seran considerados como que abandonan

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())

## --------------------Parte 4 - Evaluar, mejorar y Ajustar la RNA---------------------------

### Evaluar la **RNA** con K-Fold Validation en Keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(): #Creamos las capas de la red neuronal
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)

mean = accuracies.mean()
variance = accuracies.std()

### Mejorar la RNA
#### Regularización de Dropout para evitar el *overfitting*

### Ajustar la *RNA*
from sklearn.model_selection import GridSearchCV # sklearn.grid_search en otra version antigua de Python

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
  classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {
    'batch_size' : [25,32],
    'nb_epoch' : [100, 500], 
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

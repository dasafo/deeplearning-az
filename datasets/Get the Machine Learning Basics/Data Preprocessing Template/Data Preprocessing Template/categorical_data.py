#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:06:21 2020

@author: david
"""

#------------------------------ Datos categoricos ---------------------

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #tomamos todas las filas y todas las columnas menos la ultima
y = dataset.iloc[:, -1].values #tomamos todas las filas y la ultima columna


# Codificar datos categoricos (asociamos numeros a los paises(categoricas))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#1- Para Purchased nos vale con asociar a la variable categorica un numero cualquiera
# ya que solo hay dos posibles valores Yes(1) o No(0)
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)
# 2- Pero para los paises necesitamos la ayuda de OneHotEncoder, ya que
# asociamos las variables dummies a los paises al ser mas de 2 paises
#(y ninguno es mas que otro)
onehotencoder= OneHotEncoder(categorical_features=[0]) #culumna que queremos hacer dummie
X= onehotencoder.fit_transform(X).toarray()

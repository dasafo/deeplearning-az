#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:07:23 2020

@author: david
"""

#------------------------------ Datos faltantes ---------------------

# Tratamiento de los NAs

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3]) #fit para aplicar una funcion a un objeto
X[:, 1:3] =imputer.transform(X[:, 1:3])

# Data Preprocessing




# Dividimos el dataset en un conjunto de de training y testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0) #20% para testing, y semilla 0

# Escalado de variables (estandarizacion)

"""
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
"""
















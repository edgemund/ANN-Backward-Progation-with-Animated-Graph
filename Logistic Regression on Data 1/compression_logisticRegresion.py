############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Logistic Classification                                            #
############################################################################
from sklearn.linear_model import LogisticRegression
import numpy as np
from dataBase import selectDB
from ANN import NN

dn = "exam"   # exam , iris
# select DataBase
dataName, lableName, nClass = selectDB(dn)
NN = NN()
X, Y = NN.loadData(dataName, lableName)


# logistic Regresion
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(X, Y)
z = logreg.predict(X)
print("logistic Regression accuracy is : ",
      ((len(Y)-np.sum(np.abs(z-Y)))/len(Y))*100)
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

NN.plotBoundary(xx, yy, Z, Y, X)

############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Logistic Classification                                            #
############################################################################
import numpy as np
import matplotlib.pyplot as plt
from ANN import NN
from dataBase import selectDB

if __name__ == '__main__':
    dn = "exam"

# neural network class
NN = NN()
# select DataBase
dataName, lableName, nClass = selectDB(dn)
# load DataBase
data, lable = NN.loadData(dataName, lableName)
y = lable.reshape(-1, 1)
# plot data
NN.plotData(data, y, dn)

# initialize parameter
m, d = data.shape
epoch = 10
teta = 0.5
learningRate = 0.05

numberOfErrore = 0
accuracy = []
kCost = 0
costFunction = []
for steps in range(epoch):
    beta = data.dot(teta)
    H = NN.activationFunc(beta, "sigmoid")
    C = np.array(H > 0.5)
    H = NN.activationFunc(data, "sigmoid")
    cost = loss = (-np.sum(np.dot(y.T, np.log(H)) +
                   np.dot((1-y).T, np.log(1-H))))/m
    costFunction.append(cost)

    plt.plot(kCost, costFunction[kCost], linestyle='--', marker='o', color='g')
    plt.pause(0.1)
    kCost = kCost + 1
    if (np.sum(np.power(H-y, 2)) != 0):
        numberOfErrore = numberOfErrore + 1

    l = np.dot(data.T, (H - y)) / m
    teta -= learningRate * l

correct = m-numberOfErrore
accuracy = (((correct)/m)*100)
print("Logistic Regression Accuracy is : ", accuracy)

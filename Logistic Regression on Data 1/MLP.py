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
    #dn = input('please input Database Name: ')
    dn = "exam"


#dn = "exam"
# neural network class
NN = NN()
# select DataBase
dataName, lableName, nClass = selectDB(dn)
# load DataBase
data, y = NN.loadData(dataName, lableName)
# plot data
NN.plotData(data, y, dn)
# change lable 0 to vector [1 0 0]
lable = NN.newLable(y, nClass)
# shuffle data
data, lable = NN.shuffleData(data, lable, nClass)
# zeromean data
data = NN.zeromeanData(data)

# initialize
[m, d] = data.shape
n_fold = 5
q = 5                     # number of neuron in hidden layer
w = NN.initial(q, nClass)  # weights of output layer
v = NN.initial(d, q)       # weights of hidden layer
gama = NN.initial(q, 1)    # bias for hidden layer
teta = NN.initial(nClass, 1)    # bias for output layer
LearningRate = 0.05         # learning rate
epoch = 5
accuracy = np.zeros(n_fold)
kCost = 0
costFunction = []
for nf in range(n_fold):
    # prepare data to 5 fold cross validation
    trainData, trainLable, testData, testLable = NN.nFoldData(
        data, lable, nf, n_fold)
    [m, d] = trainData.shape

    for z in range(epoch):

        for i in range(m):
            x = trainData[i, :].reshape(-1, 1)
            y = trainLable[i].reshape(-1, 1)
#            print(y)

#           hidden layer
            alpha = np.dot(v.T, x) + gama
            alpha = alpha / np.sum(alpha, axis=0, keepdims=True)
            b = NN.activationFunc(alpha, "sigmoid")
#           output Layer
            beta = np.dot(w.T, b) + teta
            beta = beta / np.sum(beta, axis=0, keepdims=True)
            yHat = NN.activationFunc(beta, "sigmoid")
            costFunction.append(-np.sum(y*np.log(yHat))/y.shape[1])

            errorOutLayer = ((yHat - y)*yHat*(1-yHat)).reshape(-1, 1)
            errorHiddenLayer = np.zeros(q).reshape(-1, 1)
            for h in range(q):
                sumh = 0
                for j in range(nClass-1):
                    sumh += errorOutLayer[j] * w[h, j] * b[h] * (1-b[h])
                errorHiddenLayer[h] = sumh

#           update
            w = w - (LearningRate*np.dot(b, errorOutLayer.T))
            teta = teta - (LearningRate*(errorOutLayer*1))
            v = v - (LearningRate*np.dot(x, errorHiddenLayer.T))
            gama = gama - (LearningRate*(errorHiddenLayer*1))

            plt.plot(kCost, costFunction[kCost],
                     linestyle='--', marker='o', color='g')
            plt.pause(0.1)
            kCost = kCost + 1
#        print(np.mean(costFunction))

    #       test phase
    [m, d] = testData.shape
    numberOfErrore = 0
    for i in range(m):
        x = trainData[i, :].reshape(-1, 1)
        y = trainLable[i].reshape(-1, 1)

        alpha = np.dot(v.T, x) + gama
        alpha = alpha / np.sum(alpha, axis=0, keepdims=True)
        b = NN.activationFunc(alpha, "sigmoid")

        beta = np.dot(w.T, b) + teta
        beta = beta / np.sum(beta, axis=0, keepdims=True)
        yHat = NN.activationFunc(beta, "sigmoid")

        maxIndex = (yHat).argmax()
        ys = np.zeros(nClass).reshape(-1, 1)
        ys[maxIndex] = 1

        if (np.sum(np.power(ys-y, 2)) != 0):
            numberOfErrore = numberOfErrore + 1
        correct = m-numberOfErrore
    accuracy[nf] = (((correct)/m)*100)


plt.show()
plt.scatter(np.arange(len(costFunction)), costFunction, color="orange")
print("Accuracy is : ", np.mean(accuracy))

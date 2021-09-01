############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Logistic Classification                                            #
############################################################################
import numpy as np
import matplotlib.pyplot as plt


class NN():
    def __init__(self, l=2):
        self.l = l

    def loadData(self, ndata, nlable):
        newData = np.loadtxt(ndata)
        newlable = np.loadtxt(nlable, dtype=int)
        # =============================================================================
        #         newData = []
        #         i=0
        #         for line in open(ndata, 'r'):
        #             item = line.rstrip().split()
        #             floatItem = np.array(item, dtype='double')
        #             newData.append(floatItem)
        #             i = i + 1
        #         i=0
        #         newLable = []
        #         for line in open(nlable, 'r'):
        #             item = line.rstrip().split()
        #             floatItem = np.array(item, dtype='double')
        #             newLable.append(floatItem)
        # =============================================================================
        return newData, newlable

    def newLable(self, lable, nClass):
        newData = []
        for i in range(len(lable)):
            y = np.zeros(nClass)
            for j in range(nClass):
                if (lable[i] == j):
                    y[j] = 1
            newData.append(y)
        return newData

    def shuffleData(self, data, lable, nClass):
        [mm, d] = data.shape
        compactData = np.concatenate((data, lable), axis=1)
        np.random.shuffle(compactData)

        lable = compactData[:, d:d + nClass]
        clNum = []
        for i in range(nClass):
            clNum.append(d + i)

        data = np.delete(compactData, clNum, axis=1)
        return data, lable

    def zeromeanData(self, data):
        data = data - np.min(data, axis=0, keepdims=True)
        data = np.round(data / ((np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))),
                        decimals=2)
        return data

    def nFoldData(self, data, lables, nf, n_fold):
        [mm, d] = data.shape
        nFold = np.int(mm / n_fold)
        testData = data[nf * nFold:nFold * (1 + nf), :]
        testLable = lables[nf * nFold:nFold * (1 + nf), :]

        if nf != 0:
            td = data[0:nFold * nf, :]
            tl = lables[0:nFold * nf, :]
            if nf != n_fold:
                a = data[nFold * (1 + nf):mm, :]
                trainData = np.concatenate((td, a))
                trainLable = np.concatenate(
                    (tl, lables[nFold * (1 + nf):mm, :]))
            else:
                trainData = data[0:nf * nFold, :]
                trainLable = lables[0:nf * nFold, :]
        else:
            trainData = data[nFold * (1 + nf):mm, :]
            trainLable = lables[nFold * (1 + nf):mm, :]
        return trainData, trainLable, testData, testLable

    def initial(self, m, n):
        w = np.around(np.random.randn(m, n), decimals=4)
        return w

    def activationFunc(self, data, fun):
        newdata = []
        if fun == "sigmoid":
            newdata = 1 / (1 + np.exp(-data))
        return newdata

    def plotData(dself, data, lable, ndata):
        plt.figure()
        if ndata == "iris":
            index0 = np.array(
                [index for index, value in enumerate(lable) if value == 0])
            index1 = np.array(
                [index for index, value in enumerate(lable) if value == 1])
            index2 = np.array(
                [index for index, value in enumerate(lable) if value == 2])

            x1 = data[:, 0]
            x2 = data[:, 1]

            plt.scatter(x1[index0], x2[index0], color="red")
            plt.scatter(x1[index1], x2[index1], color="orange")
            plt.scatter(x1[index2], x2[index2], color="green")

        if ndata == "exam":
            index0 = np.array(
                [index for index, value in enumerate(lable) if value == 0])
            index1 = np.array(
                [index for index, value in enumerate(lable) if value == 1])

            x1 = data[:, 0]
            x2 = data[:, 1]

            plt.scatter(x1[index0], x2[index0], color="green")
            plt.scatter(x1[index1], x2[index1], color="orange")

        plt.figure(1, figsize=(10, 8))
        plt.show()

    def plotBoundary(self, xx, yy, Z, Y, X):
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(10, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        index0 = np.array(
            [index for index, value in enumerate(Y) if value == 0])
        index1 = np.array(
            [index for index, value in enumerate(Y) if value == 1])

        x1 = X[:, 0]
        x2 = X[:, 1]

        plt.figure()
        plt.scatter(x1[index0], x2[index0], color="red")
        plt.scatter(x1[index1], x2[index1], color="orange")

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()

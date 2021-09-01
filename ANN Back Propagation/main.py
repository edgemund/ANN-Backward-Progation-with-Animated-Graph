############################################################################
# ARTEFICIAL NEURAL NETWORK(ANN)                                           #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Backward Propagation                                               #
# Validation Method: 5-fold cross validation.                              #
# Author: Edmund Sowah                                                     #
############################################################################
import package
from package.module import *

print("Import data: Input 1 for Exam data, and 2 for Iris data:")
fc = int(input('Please choose data: '))
while(fc != 1 and fc != 2):
    fc = int(input('Please choose data again: '))
if fc == 1:
    pathx = 'data/exam_x.dat'
    pathy = 'data/exam_y.dat'
else:
    pathy = 'data/iris_y.dat'
    pathx = 'data/iris_x.dat'
part = int(input("Which part of test data do you want to us? Input 1,2,3,4 or 5:"))
while(part != 1 and part != 2 and
      part != 3 and part != 4 and part != 5):
    part = int(input('Please choose again: '))
X_data, y_data, c, X, y, y_test, X_test = \
    transform(pathx, pathy, part)

# Unit the first value
d0 = X.shape[0]  # Dimension of X data
d1 = h = 100  # Size of hidden layer
d2 = 10
d3 = c
loss_value, i_value = [], []
# Initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
W3 = 0.01*np.random.randn(d2, d3)
b3 = np.zeros((d3, 1))

Y = convert_labels(y, c)
N = X.shape[1]
eta = 1  # Learning rate

plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for i in range(1000):
    # Feedforward
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = np.dot(W3.T, A2) + b3
    Yhat = softmax(Z3)

    # Print loss after each 1000 iterations
    if i % 10 == 0:
        # Calculate the loss: average cross-entropy loss
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" % (i, loss))
        loss_value.append(loss)
        i_value.append(i)
        # Display loss function
        # ax[0].cal()
        # show All Data
        plt.subplot(1, 3, 1)
        plt.title("Ground truth")
        plt.xlabel("X1")
        plt.ylabel("X2")
        for i in range(y_data.shape[0]):
            if y_data[i] == 0:
                plt.plot(X_data[i, 0], X_data[i, 1], 'go')
            if y_data[i] == 1:
                plt.plot(X_data[i, 0], X_data[i, 1], 'r*')
            if y_data[i] == 2:
                plt.plot(X_data[i, 0], X_data[i, 1], 'b^')
        # show training
        plt.subplot(1, 3, 2)
        plt.title("Classification scatter plot")
        plt.xlabel("X1")
        plt.ylabel("X2")
        predicted = np.argmax(Z3, axis=0)
        for i in range(predicted.shape[0]):
            if predicted[i] == 0:
                plt.plot(X_data[i, 0], X_data[i, 1], 'go')
            if predicted[i] == 1:
                plt.plot(X_data[i, 0], X_data[i, 1], 'ro')
            if predicted[i] == 2:
                plt.plot(X_data[i, 0], X_data[i, 1], 'bo')
        # Show loss
        plt.subplot(1, 3, 3)
        plt.title("Loss value")
        plt.xlabel("Number loops")
        plt.ylabel("Loss value")
        # ax[1].cla()
        plt.plot(i_value, loss_value, 'go-')
        plt.pause(0.01)

    # Back Propagation
    E3 = (Yhat - Y)/N
    dW3 = np.dot(A2, E3.T)
    db3 = np.sum(E3, axis=1, keepdims=True)
    E2 = np.dot(W3, E3)
    E2[Z2 <= 0] = 0  # gradient of ReLU
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis=1, keepdims=True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0  # gradient of ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis=1, keepdims=True)
    # Gradient Descent update
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2
    W3 += -eta*dW3
    b3 += -eta*db3

# use data testing
Z1 = np.dot(W1.T, X_test) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
A2 = np.maximum(Z2, 0)
Z3 = np.dot(W3.T, A2) + b3
predicted_class = np.argmax(Z3, axis=0)
print("Predicted class:\n", predicted_class)
print("Real value:\n", y_test)
print('Training Accuracy: %.2f %%'
      % (100*np.mean(predicted_class == y_test)))

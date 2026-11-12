#
# Template for Task 1: Linear Regression
#

import matplotlib.pyplot as plt
import numpy as np

# --- Your Task --- #
# import libraries as needed
from random import random
# --- end of task --- #

# -------------------------------------
# load data
data = np.loadtxt("crimerate.csv", delimiter=",")
[n, p] = np.shape(data)
# 75% for training, 25% for testing
num_train = int(0.75 * n)
num_test = int(0.25 * n)
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]
sample_test = data[n - num_test :, 0:-1]
label_test = data[n - num_test :, -1]
# -------------------------------------


# --- Your Task --- #
# pick a proper number of iterations
num_iter = 1000
# randomly initialize your w
w = np.random.rand(p) * 0.01

learning_rate = 0.01

x_train = np.hstack([np.ones((num_train, 1)), sample_train])
x_test = np.hstack([np.ones((num_test, 1)), sample_test])
# --- end of task --- #

er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w
for iter in range(num_iter):
    ## update w
    y_hat = x_train @ w
    w += learning_rate * x_train.T @ (label_train - y_hat) / num_train

    ## evaluate testing error of the updated w
    # we should measure mean-square-error here
    er = np.mean((x_test @ w - label_test) ** 2)
    er_test.append(er)
# --- end of task --- #

plt.figure()
plt.plot(er_test)
plt.xlabel("Iteration")
plt.ylabel("Mean Square Error")
plt.show()

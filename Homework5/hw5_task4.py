#
# Template for Task 4: kNN Classification
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed
# .......
# --- end of task --- #

# -------------------------------------
# load data
data = np.loadtxt('datasets/diabetes.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Your Task --- #
# pick five values of k by yourself
k_values = [1, 5, 15, 50, 200]
# --- end of task --- #

er_test = []
for k in k_values:
    # --- Your Task --- #
    # implement the kNN classification method
    preds = []
    for x in sample_test:
        dists = np.linalg.norm(sample_train - x, axis=1)
        nn = np.argsort(dists)[:k]
        preds.append(1 if np.mean(label_train[nn]) >= 0.5 else 0)
    preds = np.array(preds)

    # store classification error on testing data here
    er = np.mean(preds != label_test)
    er_test.append(er)
# --- end of task --- #

plt.figure()
plt.plot(k_values, er_test)
plt.xlabel('k')
plt.ylabel('Classification Error')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# --- end of task --- #

# load an imbalanced data set
# there are 50 positive class instances
# there are 500 negative class instances
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

per = 0.5
num_train = int(n * per)
sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]

auc_per = []

weights = [1, 2, 5, 10, 20, 50]
for weight in weights:

    model = LogisticRegression(class_weight={0: 1, 1: weight})

    model.fit(sample_train, label_train)

    prob_base = model.predict_proba(sample_test)[:, 1]
    auc = roc_auc_score(label_test, prob_base)
    auc_per.append(auc)


plt.figure()
plt.plot(weights,auc_per)
plt.xlabel('Positive Class Weight')
plt.ylabel('AUC Score')
plt.savefig('figure6.png')

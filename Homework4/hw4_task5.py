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

# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per:

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains
    # the model using sample_train and label_train
    model.fit(sample_train, label_train)

    # evaluate model testing accuracy and stores it in "acc_base"
    acc_base = model.score(sample_test, label_test)
    acc_base_per.append(acc_base)

    # evaluate model testing AUC score and stores it in "auc_base"
    prob_base = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, prob_base)
    auc_base_per.append(auc_base)
    # --- end of task --- #


    # --- Your Task --- #
    # Now, implement your method
    # Aim to improve AUC score of baseline
    # while maintaining accuracy as much as possible
    model = LogisticRegression(class_weight='balanced')
    model.fit(sample_train, label_train)

    # evaluate model testing accuracy and stores it in "acc_yours"
    acc_yours = model.score(sample_test, label_test)
    acc_yours_per.append(acc_yours)

    # evaluate model testing AUC score and stores it in "auc_yours"
    prob_yours = model.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, prob_yours)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #


plt.figure()
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.savefig('figure4.png')

plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.savefig('figure5.png')

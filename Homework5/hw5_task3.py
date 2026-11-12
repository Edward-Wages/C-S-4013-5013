#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed
# .......
# --- end of task --- #

# -------------------------------------
# load data
# note we do not need label
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters
k = 3
# --- end of task --- #


# --- Your Task --- #
# implement the Kmeans clustering algorithm
# you need to first randomly initialize k cluster centers
rng = np.random.default_rng(0)
centers = sample[rng.choice(n, k, replace=False)]   # random init from data points
label_cluster = np.zeros(n, dtype=int)
for _ in range(100):
    dist = np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2)
    new_label = np.argmin(dist, axis=1)

    new_centers = np.array([sample[new_label == j].mean(axis=0) if np.any(new_label == j)
                            else centers[j] for j in range(k)])
    if np.array_equal(new_label, label_cluster) and np.allclose(new_centers, centers):
        break
    label_cluster, centers = new_label, new_centers
# --- end of task --- #


# the following code plot your clustering result in a 2D space
pca = PCA(n_components=2)
pca.fit(sample)
sample_pca = pca.transform(sample)
idx = []
colors = ['blue','red','green','m']
for i in range(k):
     idx = np.where(label_cluster == i)
     plt.scatter(sample_pca[idx,0],sample_pca[idx,1],color=colors[i],facecolors='none')

plt.show()

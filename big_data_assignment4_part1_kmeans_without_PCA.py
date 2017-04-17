#!/usr/bin/env python2
#-*- coding: utf-8 -*-
##########################K means without PCA##################

# Modified based on Gaël Varoquaux's code by jimmy shen on April, 2017.
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets

#This function can compare the clustering result with the ground truth
# and then return the correct clustering ratio.
def cluster_result_analysis(a, b, class_number):
    a=list(a)
    b=list(b)
    correct_cluster=0
    for j in range(class_number):
        cluster_result_of_the_ground_truth=[]
        indexes = [i for i,x in enumerate(a) if x == j]
        for k in range(len(indexes)):
            cluster_result_of_the_ground_truth.append(b[indexes[k]])
        print "The cluster result of class",j, ":\n",cluster_result_of_the_ground_truth
        number_of_correct_cluster=cluster_result_of_the_ground_truth.count(max(cluster_result_of_the_ground_truth,key=cluster_result_of_the_ground_truth.count))
        print "#correct clustering:",number_of_correct_cluster
        correct_cluster+=number_of_correct_cluster
    return float(correct_cluster)/len(b)



#####   Plot the clustering result   ########################
np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target
#print("X", X)
#print(type(y))
print "ground truth label\n", y
estimators = {'k_means_iris_3_without_PCA': KMeans(n_clusters=3),}


for name, est in estimators.items():
    
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    plt.cla()
    est.fit(X)
    labels = est.labels_
    print "clustering labels\n",labels
    print "The K mean clustering Accuracy without PCA:%.4f" % (cluster_result_analysis(y,labels,3))
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(label=name, fontdict=None, loc=u'center')



#####   Plot the ground truth   ########################
fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
plt.cla()


for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 0, 2]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title(label="Ground Truth", fontdict=None, loc=u'center')
plt.show()

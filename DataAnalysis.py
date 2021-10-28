from sklearn.decomposition import PCA
import numpy as np
import math
import scipy.cluster as sc
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from numpy import unique
from numpy import where
from sklearn.cluster import Birch
from matplotlib.colors import ListedColormap
from  sklearn.model_selection  import  train_test_split
from  sklearn.preprocessing  import  StandardScaler
from  sklearn.neighbors  import (KNeighborsClassifier ,NeighborhoodComponentsAnalysis)
from  sklearn.pipeline  import  Pipeline
from sklearn import svm
from itertools import cycle
import time 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# PRE PROCESSING 

def normalise(data):
    normalisedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        maxElement = np.amax(data[:,j])
        minElement = np.amin(data[:,j])
        
        for i in range(rows):
            normalisedData[i,j] = (data[i,j] - minElement) / (maxElement - minElement)
            
    return normalisedData

def normalise2(data):
    normalisedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        maxElement = np.amax(data[:,j])
        minElement = np.amin(data[:,j])
        
        for i in range(rows):
            normalisedData[i,j] = -1 + (data[i,j] - minElement) * (1 - (-1)) / (maxElement  - minElement)
            
    return normalisedData

def standard(data):
    standardData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        sigma = np.std(data[:,j])
        mu = np.mean(data[:,j])
        
        for i in range(rows):
            standardData[i,j] = (data[i,j] - mu)/sigma
            
    return standardData

def centralize(data):
    centralizedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        mu = np.mean(data[:,j])

        for i in range(rows):
            centralizedData[i,j] = (data[i,j] - mu)

    return centralizedData

def dist(p1, p2):
    sumTotal = 0

    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]),2)

    return math.sqrt(sumTotal)



dataRaw = []
DataFile = open("SCC403CWWeatherData.txt", "r")

while True:
    theline = DataFile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos])
    dataRaw.append(readData)
    
DataFile.close()

data = np.array(dataRaw)

standardData = standard(data)

normalisedData = normalise(standardData)
    
normalisedData2 = normalise2(standardData)

centralizedData = centralize(normalisedData)

# NORMALISED DATA PLOTS
plt.figure(figsize=(6,4))
# Temperature / Humidity
plt.scatter(normalisedData2[:,2], normalisedData2[:,5], s=5)
plt.xlabel('Mean Temperature')
plt.ylabel('Mean Humidity')
plt.show()
plt.close()

# PCA

pca = PCA(n_components=2)
pca1 = pca.fit(centralizedData)
Coeff = pca.components_

transformedData = pca1.transform(centralizedData)

plt.figure(figsize=(6,4))
plt.plot(transformedData[:,0], transformedData[:,1], ".")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.title("PCA")
plt.savefig("PCAData.png")
plt.show()

print("PCA DATA:")
print("explained variance ratio:")
print(pca.explained_variance_ratio_)
print("explained variance:")
print(pca.explained_variance_)
print("estimating the covariance matrix of the reduced space:")
print(np.cov(transformedData.T))
print("Feature Importance")
print(abs( pca.components_ ))

# CLUSTERING

# K means

reducedData = data[:,2:5]
normalisedData2 = normalise2(reducedData)

centroids, distortion = sc.vq.kmeans(normalisedData2 , 4)

plt.figure(figsize=(6,4))

plt.plot(normalisedData2 [:,0],normalisedData2 [:,1],'.')

plt.plot(centroids[0,0],centroids[0,1],'rx')

plt.plot(centroids[1,0],centroids[1,1],'gx')

plt.xlabel('Mean Temperature')
plt.ylabel('Mean Humidity')
plt.title("K Means")
plt.savefig("kmeans.png")
plt.show()

plt.close()

group1 = np.array([])

group2 = np.array([])

for d in normalisedData2 :
    if (dist(d, centroids[0,:]) < dist(d, centroids[1,:])):
        if (len(group1) == 0):
            group1 = d
        else:
            group1 = np.vstack((group1,d))
    else:
        if (len(group2) == 0):
            group2 = d
        else:
            group2 = np.vstack((group2,d))

plt.figure(figsize=(6,4))

plt.plot(group1[:,0],group1[:,1],'r.')
plt.plot(group2[:,0],group2[:,1],'g.')

plt.plot(centroids[0,0],centroids[0,1],'rx')
plt.plot(centroids[1,0],centroids[1,1],'gx')

plt.xlabel('Mean Temperature')
plt.ylabel('Mean Humidity')
plt.title("K Means classified")
plt.savefig("kmeansClassifiedClustering.png")
plt.show()


plt.close()


# MeanShift

reducedData = data[:,2:5]
normalisedData2 = normalise2(reducedData)

bandwidth = estimate_bandwidth(normalisedData2, quantile=0.2)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(normalisedData2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(normalisedData2[my_members, 0], normalisedData2[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    
plt.xlabel('Mean Temperature')
plt.ylabel('Mean Humidity')
plt.title("Mean shift clustering")
plt.savefig("MeanShiftClustering.png")
plt.show()


# Birch 

reducedData = data[:,2:5]

normalisedData2 = normalise2(reducedData)

# define the model
model = Birch(threshold=0.01, n_clusters=2)
# fit the model
model.fit(normalisedData2)
# assign a cluster to each example
yhat = model.predict(normalisedData2)
# retrieve unique clusters
clusters = unique(yhat)
plt.figure(figsize=(6,4))
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(normalisedData2[row_ix, 0], normalisedData2[row_ix, 1], s=5)
# show the plot
plt.xlabel('Mean Temperature')
plt.ylabel('Mean Humidity')
plt.title("Birch clustering")
plt.savefig("BirchClustering.png")
plt.show()



# CLASSIFICATION


# KNN
starttime = time.time()
n_neighbors = 3

X = normalise2(data)

# Only take the mean temperature and humidity readings
X = X[:, [2, 5]]

y = []

# make target groups
position = 0
for i in X:
    
    if X[position, 0] > 0.0 and X[position, 1] > 0: # warm and humid
        y.append(0)
    elif X[position, 0] > 0.0 and X[position, 1] < 0: # warm and not humid
        y.append(1)
    elif X[position, 0] < 0.0 and X[position, 1] > 0: # cold and humid
        y.append(2)
    elif X[position, 0] < 0.0 and X[position, 1] < 0: # cold and nit humid
        y.append(3)
    position = position + 1

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size=0.8, random_state=42)

h = .01  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['red', 'green', 'blue', 'orange'])
cmap_bold = ListedColormap(['red', 'green', 'blue', 'orange'])

names = ['KNN', 'NCA, KNN']

classifiers = [Pipeline([('scaler', StandardScaler()),
                         ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
                         ]),
               Pipeline([('scaler', StandardScaler()),
                         ('nca', NeighborhoodComponentsAnalysis()),
                         ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
                         ])
               ]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=.8)

    # Plot also the training and testing points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("{} (k = {})".format(name, n_neighbors))
    plt.text(0.9, 0.1, '{:.2f}'.format(score), size=15,
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel("Mean Temperature")
    plt.ylabel("Mean Humidity")
    plt.savefig("KNNClassification" + name + ".png")
    plt.show()

    y_pred = clf.predict(X_test)

    print("\n" + "Classification Report for " + name)
    print(classification_report(y_test, y_pred))
    
    endtime = time.time() - starttime
    print(f"Time taken {endtime}")
    starttime = time.time()


# SVM
starttime = time.time()

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


X = normalise2(data)
# Only take the mean temperature and humidity readings
X = X[:, [2, 5]]

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
    
for clf, title in zip(models, titles):
    plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Mean Temperature")
    plt.ylabel("Mean Humidity")
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.savefig("SVMClassification" + title + ".png")
    plt.show()
    
    y_pred = clf.predict(X_test)
    print("\n" + "Classification Report for " + title)
    print(classification_report(y_test, y_pred))

    endtime = time.time() - starttime
    print(f"Time taken {endtime}")
    starttime = time.time()
import pandas as pd 
import numpy as np
wine = pd.read_csv("D:\\excelR\\Data science notes\\PCA reduction\\py\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine.data = wine.ix[:,1:]
wine.data.head(4)
wine.data
# Normalizing the numerical data 
wine_normal = scale(wine.data)

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine_normal)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:3]
plt.scatter(x,y,color=["red"])




################### Clustering  ##########################
k_df = pd.DataFrame(pca_values[:,0:3])

# K-means clustering
# Using the Elbow Method to find optimal no.of clusters
wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
	kmeans.fit(k_df)
	wcss.append(kmeans.inertia_)
# scree plot
plt.plot(range(1,11), wcss, 'ro-');plt.title('The Elbow Method');plt.xlabel('No.of Clusters');plt.ylabel('wcss')

# Apply K-Means to the crime dataset [4 clusters i get]
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(k_df)

# Getting Labels of Clusters assigned to each row
kmeans.labels_

k_df['clusters'] = kmeans.labels_ 
k_df.head()

k_df = k_df.iloc[:,[3,0,1,2]]
k_df.head()

k_groups = k_df.iloc[:,1:].groupby(k_df.clusters).median()
k_groups

# H-clustering
h_df = pd.DataFrame(pca_values[:,0:3])

# Using Dendrogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(h_df, method='ward'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distance')
# i get 5 clusters
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(h_df)

# Getting Labels of Clusters assigned to each row
hc.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
h_df['clusters'] = hc.labels_ # creating a new column and assigning it to new column
h_df.head()

h_df = h_df.iloc[:,[3,0,1,2]]

h_groups = h_df.iloc[:,1:].groupby(h_df.clusters).median()
h_groups

# clustering on original data
knorm_data = pd.DataFrame(wine_normal)
knorm_data.shape

# K-means clustering
# Using the Elbow Method to find optimal no.of clusters
wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
	kmeans.fit(knorm_data)
	wcss.append(kmeans.inertia_)
# scree plot
plt.plot(range(1,11), wcss, 'ro-');plt.title('The Elbow Method');plt.xlabel('No.of Clusters');plt.ylabel('wcss')

# Apply K-Means to the crime dataset [4 clusters i get]
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y1_kmeans = kmeans.fit_predict(knorm_data)

# Getting Labels of Clusters assigned to each row
kmeans.labels_

knorm_data['clusters'] = kmeans.labels_ 
knorm_data.head()

knorm_data = knorm_data.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
knorm_data.head()

knorm_groups = knorm_data.iloc[:,1:].groupby(knorm_data.clusters).median()
knorm_groups

# H-clustering
hnorm_data = pd.DataFrame(wine_normal)

# Using Dendrogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(hnorm_data, method='complete'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distance')
# i get 4 clusters
# Fitting Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
y1_hc = hc.fit_predict(hnorm_data)

hc.labels_
hnorm_data['clusters'] = hc.labels_  
hnorm_data.head()

h_groups = hnorm_data.iloc[:,1:].groupby(hnorm_data.clusters).median()
h_groups

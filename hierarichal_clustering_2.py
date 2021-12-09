# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 01:05:18 2021

@author: ethan
"""

import pandas as pd
import numpy as np
import sklearn
from pyclustertend import hopkins
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("second_dataset.csv")
X = data.iloc[:, :21]
X = X.drop("fetal_movement",axis= 1)
X_data = X
y =  data['fetal_health']

X = scale(X)

y= y.map({1: "Normal", 2: "Suspect", 3: "Pathological" })

# %% Heirarical Clustering on all data
sns.clustermap(X, method='ward')
sns.clustermap(X, method = 'complete')
sns.clustermap(X, method = 'average')

# %% Heirarical Clustering on only pathological

X_p = X[y=='Pathological', :]
sns.clustermap(X_p, method='ward')
sns.clustermap(X_p, method = 'complete')
sns.clustermap(X_p, method = 'average')

# %% 
#Tsne
from sklearn.manifold import TSNE
X_tsne = TSNE(random_state=101).fit_transform(X)

x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
sns.scatterplot(x= x, y = y2, hue =y )

# %%
from sklearn.cluster import AgglomerativeClustering

fig, axs = plt.subplots(ncols=4)

links = ['single', 'complete', 'average', 'ward']
plt.title("Hierarichal Clustering Projected on Tsne")
for i, link in enumerate(links):
    agg = AgglomerativeClustering(n_clusters = 2, linkage = link)
    agg.fit(X_p)
    labels = agg.labels_
    X_tsne = TSNE(random_state=105).fit_transform(X[y=='Pathological', :])
    x_p = np.array(X_tsne[:,0])
    y_p = np.array(X_tsne[:,1])
    sns.scatterplot(x= x_p, y = y_p, hue = labels.astype(str), ax= axs[i])
    axs[i].title.set_text(link)
plt.show()


# %% Absolute Difference of Cluster feature means

agg = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
agg.fit(X_p)
labels = agg.labels_

X1 = X_p[labels == 0]
X2 = X_p[labels == 1]

diffs = abs(X1.mean(0) - X2.mean(0))

diffs = pd.Series(diffs, index = X_data.columns).sort_values()
diffs.plot.barh(y='ABS of Means')
plt.title("Abs Diff between Cluster Feature Means")
plt.tight_layout()

# %% Post Clustering Selection

feats = list(diffs[-2:].index)
X_2 = np.array(X_data[feats])
X_2 = X_2[y=='Pathological', :]
X_2 = scale(X_2)

agg = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
agg.fit(X_2)
labels = agg.labels_

#Tsne
from sklearn.manifold import TSNE
X_2_tsne = TSNE(random_state=101).fit_transform(X_2)

t1 = np.array(X_2_tsne[:,0])
t2 = np.array(X_2_tsne[:,1])
sns.scatterplot(x= t1, y = t2 , hue=labels)

# %% Post Clustering Selection

feats = list(diffs[-2:].index)
feats = ['percentage_of_time_with_abnormal_long_term_variability',diffs[-1:].index[0] ]

X_2 = np.array(X_data[feats])
X_2 = X_2[y=='Pathological', :]
X_2 = scale(X_2)

agg = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
agg.fit(X_2)
labels = agg.labels_

#Tsne
from sklearn.manifold import TSNE
X_2_tsne = TSNE(random_state=101).fit_transform(X_2)

t1 = X_2[:,0]
t2 = X_2[:,1]
plt.xlabel(feats[0])
plt.ylabel(feats[1])
sns.scatterplot(x=t1, y = t2 , hue=labels)

#%%

sns.clustermap(X_data.corr())

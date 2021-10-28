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

data = pd.read_csv("fetal_health.csv")
X = data
X, y = data.iloc[:,:-1], data['fetal_health'] 
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
    agg = AgglomerativeClustering(n_clusters = 3, linkage = link)
    agg.fit(X_p)
    labels = agg.labels_
    X_tsne = TSNE(random_state=105).fit_transform(X[y=='Pathological', :])
    x_p = np.array(X_tsne[:,0])
    y_p = np.array(X_tsne[:,1])
    sns.scatterplot(x= x_p, y = y_p, hue = labels.astype(str), ax= axs[i])
    axs[i].title.set_text(link)
plt.show()
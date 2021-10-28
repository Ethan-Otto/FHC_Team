# -*- coding: utf-8 -*-
"""1

@author: ethan
"""

import pandas as pd
import numpy as np
import sklearn
from pyclustertend import hopkins
from sklearn.preprocessing import scale
import seaborn as sns

data = pd.read_csv("fetal_health.csv")
X = data
X, y = data.iloc[:,:-1], data['fetal_health'] 
X = scale(X)

y= y.map({1: "Normal", 2: "Suspect", 3: "Pathological" })
hop = hopkins(X,300)
print(f'General Hopkins test {hop}' )

hop1 = hopkins(X[y=="Normal", :],100)
print(f'Normal Fetus Hopkins test {hop1}' )

hop2 = hopkins(X[y=="Suspect", :],100)
print(f'Suspect Hopkins test {hop2}' )

hop3 = hopkins(X[y=="Pathological", :],100)
print(f'Pathlological Hopkins test {hop3}' )


# %%

#Tsne
from sklearn.manifold import TSNE
X_tsne = TSNE(random_state=101).fit_transform(X)

x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
sns.scatterplot(x= x, y = y2, hue =y )

# %% 

X_tsne = TSNE(random_state=105).fit_transform(X[y=='Suspect', :])
x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
sns.scatterplot(x= x, y = y2 )

# %% 

X_tsne = TSNE(random_state=105).fit_transform(X[y=='Pathological', :])
x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
sns.scatterplot(x= x, y = y2 )




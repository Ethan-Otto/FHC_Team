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
import matplotlib as plt

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

X_tsne = TSNE(random_state=100).fit_transform(X[y=='Suspect', :])
x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
plt.title("TSNE- Suspect")
sns.scatterplot(x= x, y = y2 )

# %% 

X_tsne = TSNE(random_state=100).fit_transform(X[y=='Normal', :])
x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
plt.title("TSNE Normal")
sns.scatterplot(x= x, y = y2 )


# %% 

X_tsne = TSNE(random_state=105).fit_transform(X[y=='Pathological', :])
x = np.array(X_tsne[:,0])
y2 = np.array(X_tsne[:,1])
plt.title("TSNE Pathological")
sns.scatterplot(x= x, y = y2 )

# %% New Dataset


data2 = pd.read_csv("second_dataset.csv")
X2 = data2.iloc[:, :21]
X2 = X2.drop("fetal_movement",axis= 1)
y2 =  data2['fetal_health']

X2 = scale(X2)


y2= y2.map({1: "Normal", 2: "Suspect", 3: "Pathological" })
hop = hopkins(X2,300)
print(f'General Hopkins test {hop}' )

hop1 = hopkins(X2[y2=="Normal", :],100)
print(f'Normal Fetus Hopkins test {hop1}' )

hop2 = hopkins(X2[y2=="Suspect", :],100)
print(f'Suspect Hopkins test {hop2}' )

hop3 = hopkins(X2[y2=="Pathological", :],100)
print(f'Pathlological Hopkins test {hop3}' )



#%% Tsne
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X2_tsne = TSNE(random_state=101).fit_transform(X2)


x = np.array(X2_tsne[:,0])
x2 = np.array(X2_tsne[:,1])
sns.scatterplot(x= x, y = x2, hue =y2 )
plt.title("TSNE- All Classes")

# %% 

X_tsne = TSNE(random_state=100).fit_transform(X2[y2=='Suspect', :])
x = np.array(X_tsne[:,0])
x2 = np.array(X_tsne[:,1])

sns.scatterplot(x= x, y = x2)
plt.title("TSNE- Suspect")

# %% 

X_tsne = TSNE(random_state=100).fit_transform(X2[y2 =='Normal', :])
x = np.array(X_tsne[:,0])
x2 = np.array(X_tsne[:,1])
plt.title("TSNE Normal")
sns.scatterplot(x= x, y = x2 )


# %% 

X_tsne = TSNE(random_state=105).fit_transform(X2[y2=='Pathological', :])
x = np.array(X_tsne[:,0])
x2 = np.array(X_tsne[:,1])
plt.title("TSNE Pathological")
sns.scatterplot(x= x, y = x2 )

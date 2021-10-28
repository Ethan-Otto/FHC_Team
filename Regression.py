# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:44:29 2021

https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
"""

from numpy import mean
from numpy import std
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

data = pd.read_csv("fetal_health.csv")
X, y = data.iloc[:,:-1], data['fetal_health'] 



#%% 
 
# get a list of models to evaluate
def get_models(m):
    models = dict()
    if m == type(LogisticRegression()):
        for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
           		# create name for model
           		key = '%.4f' % p
           		# turn off penalty in some cases
           		if p == 0.0:
           			# no penalty in this case
           			models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
           		else:
           			models[key] = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', penalty='l2', C=p)  
    else:
        #Number of tree estimators
        for p in [100,500,1000]:
            key = '%.4f' % p
            models[key] = RandomForestClassifier(n_estimators = p)
    return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# define dataset
# get the models to evaluate
models = get_models(RandomForestClassifier())
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model and collect the scores
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize progress along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()



# %% Testing on individual Random Forest
rfc = RandomForestClassifier(n_estimators = 500)

X_train, X_test, y_train, y_test = train_test_split(X, y)
rfc.fit(X_train, y_train)
fi = rfc.feature_importances_
# %%
#Ploting the rf feature importances
fig, ax = pyplot.subplots()
fi = pd.Series(fi, index = X.columns).sort_values()
fi.plot.barh(ax=ax)
fig.tight_layout()
import pandas as pd
import numpy as np
from drug_consumption.clean_data import canna_df
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

print("Shape of Cannabis data:", canna_df.shape)

print("Counts per response:\n", {
    n: v for n, v in zip(canna_df.Cannabis.value_counts().index, canna_df.Cannabis.value_counts())
})

# We won't one-hot encode the age feature because it is an ordinal feature. 
# We can maintain more information about the feature by creating a new feature for the lower bound of the age and the upper bound the age.
# This strategy maintains more information about the age than a simple mean.
canna_df['Lower_Age'], canna_df['Upper_Age'] = zip(*canna_df['Age'].map(lambda x: x.split('-')))

# Make dataframe of features
canna_features=canna_df.drop(['ID','Age', 'Cannabis'], axis = 1)
canna_features.Lower_Age=canna_features.Lower_Age.astype('float')
canna_features.Upper_Age=canna_features.Upper_Age.astype('float')

canna_df.Cannabis.value_counts() / len(canna_df)
# Because 64% of responses are in class 1, there would be a baseline accuracy of 0.64 if we predicted all responses to be class 1
# Because this data is slightly imbalanced, ROC-AUC may be a better evaluation metric than accuracy.  
# Area under the ROC curve shows the probability that a value of the positive class will have a higher score than one of the negative class according to the decision function of the model

# Split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(canna_features, canna_df.Cannabis, random_state= 42)

y_train.value_counts() / len(y_train)


# Negative class is the first entry, ie. 0
grid_search.classes_



# KNN

# Logistic Regression

# SVC

ct = make_column_transformer(
    (['Upper_Age', 'Lower_Age'], StandardScaler()), 
     (['Gender', 'Education', 'Country', 'Ethnicity'], OneHotEncoder(sparse=False)), 
     remainder='passthrough')
    
ct.fit(X_train)
X_train_trans=ct.transform(X_train)
X_test_trans=ct.transform(X_test)


# Make parameter grid for grid search 
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100], 'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
print("parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_trans, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-val accuracy score: {:.2f}".format(grid_search.best_score_))

print("Test set roc-auc: {:.2f}".format(roc_auc_score(y_test, grid_search.decision_function(X_test_trans))))
# roc-auc 0.81
print("Test set accuracy score: {:.2f}".format(grid_search.score(X_test_trans, y_test)))
# accuracy 0.76


results = pd.DataFrame(grid_search.cv_results_)
results.head()
scores = np.array(results.mean_test_score).reshape(6,6)

heatmap=sns.heatmap(scores, annot=True)
plt.xlabel('gamma')
plt.ylabel('C')
plt.xticks(np.arange(6), param_grid['gamma'])
plt.yticks(np.arange(6), param_grid['C'])



# Use AUC scoring instead
grid_search_auc = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True, scoring='roc_auc')
grid_search_auc.fit(X_train_trans, y_train)

print("Test set roc-auc: {:.2f}".format(roc_auc_score(y_test, grid_search_auc.decision_function(X_test_trans))))
print("Test set accuracy score: {:.2f}".format(accuracy_score(y_test, grid_search_auc.predict(X_test_trans))))


print("Best parameters: {}".format(grid_search_auc.best_params_))
print("Best cross-val roc-auc: {:.2f}".format(grid_search_auc.best_score_))

results = pd.DataFrame(grid_search_auc.cv_results_)
results.head()
scores = np.array(results.mean_test_score).reshape(6,6)

heatmap=sns.heatmap(scores, annot=True)
plt.xlabel('gamma')
plt.ylabel('C')
plt.xticks(np.arange(6), param_grid['gamma'])
plt.yticks(np.arange(6), param_grid['C'])


# Adjust parameter grid for search, and also search the linear kernel


param_grid = [
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}, 
    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train_trans, y_train)
print("Test set score: {:.2f}".format(grid_search.score(X_test_trans, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-val score: {:.2f}".format(grid_search.best_score_))
print("Best estimator: \n{}".format(grid_search.best_estimator_))

results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results[results.param_kernel=='linear'].mean_test_score).reshape(6,1)

heatmap=sns.heatmap(scores, annot=True)
plt.ylabel('C')
plt.yticks(np.arange(6), results[results.param_kernel=='linear'].param_C)

# This seems like a better parameter grid space because the plot is filled up with evenly with the higher cross-val scores


# AUC scoring

grid_search_auc = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True, scoring='roc_auc')
grid_search_auc.fit(X_train_trans, y_train)

print("Test set roc-auc: {:.2f}".format(roc_auc_score(y_test, grid_search_auc.decision_function(X_test_trans))))
print("Test set accuracy score: {:.2f}".format(accuracy_score(y_test, grid_search_auc.predict(X_test_trans))))
print("Best parameters: {}".format(grid_search_auc.best_params_))
print("Best cross-val roc-auc: {:.2f}".format(grid_search_auc.best_score_))


results = pd.DataFrame(grid_search_auc.cv_results_)
results.head()
scores = np.array(results[results.param_kernel=='rbf'].mean_test_score).reshape(6,6)

heatmap=sns.heatmap(scores, annot=True)
plt.xlabel('gamma')
plt.ylabel('C')
plt.xticks(np.arange(6), results[results.param_kernel=='rbf'].param_gamma)
plt.yticks(np.arange(6), results[results.param_kernel=='rbf'].param_C)


# Next we'll take a look at the ROC curve to see if the threshold of the decision function is in the optimal place

# This plots the false positive rate by the true positive rate for every threshold value in the decision curve

fpr, tpr, thresholds = roc_curve(y_test, grid_search.decision_function(X_test_trans))
thresh = np.argmin(np.abs(thresholds))

plt.plot(fpr, tpr, label="ROC Curve")
plt.ylabel('FPR (recall)')
plt.xlabel('TPR')
plt.plot(fpr[thresh], tpr[thresh], 'o', markersize = 10, label = "threshold zero")
plt.legend(loc=4)


# Looks like the default threshold for the decision function is in a pretty good spot: it balances a high TPR and high FPR





# Random forest




from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
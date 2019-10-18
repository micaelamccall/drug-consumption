from drug_consumption.prepare_data import X_train, y_train
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from drug_consumption.wrapper import grid_search_wrapper


# Make the preprocessor 
ct = ColumnTransformer([
    ("num", StandardScaler(), ['Upper_Age', 'Lower_Age']), 
     ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse=False), ['Gender', 'Education', 'Country', 'Ethnicity'])], 
     remainder='passthrough')


pipe = Pipeline([
    ("preparation", ct),
    ("svc", SVC())
])


# Make list of scorers to evaluate predictions
scorers = {
    'roc_auc' : make_scorer(roc_auc_score),
    'precision_score' : make_scorer(precision_score),
    'accuracy' : make_scorer(accuracy_score)
}


if __name__ == "__main__":
    # Make parameter grid for grid search 

    param_grid = [
        {'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}, 
        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
    
    print("parameter grid:\n{}".format(param_grid), "\n\nscorers to evaluate predictions on test set:\n{}".format(list(scorers.keys())))

    grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='roc_auc')
    grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='accuracy')



# Because the largest scores are near the endge of the heat map, we'll adjust the parameter grid to see if better scores are beyong the search space of the last grid search
param_grid = [
    {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100, 1000, 10000], 'svc__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}, 
    {'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search, svc_results = grid_search_wrapper(X_train, y_train, pipe, param_grid, refit_score='roc_auc')
# This seems like a better parameter grid space because the plot is filled up with evenly with the higher cross-val scores
# But the best parameters are the same as the ones we found in the first grid search

svc_pipe = grid_search.best_estimator_

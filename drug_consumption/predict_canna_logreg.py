from drug_consumption.prepare_data import X_train, y_train, X_test, y_test
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from drug_consumption.test import grid_search_wrapper

# Column transformer to one-hot encode categorical features and scale numerical ones
ct = ColumnTransformer([
    ("num", StandardScaler(), ['Upper_Age', 'Lower_Age']), 
     ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse=False), ['Gender', 'Education', 'Country', 'Ethnicity'])], 
     remainder='passthrough')


pipe = Pipeline([
    ("preparation", ct),
    ("logreg", LogisticRegression())
])

# l1 and l2 are types of regularization; they restrict the model and prevent it from overfitting by restricting the coefficients to be near zero. 
# l1 also forces some to be exactly 0 so that they can be ignorned entirely

param_grid = {
    'logreg__C': [0.001, 0.01, 1, 100],
    'logreg__penalty': ["l1", "l2"]
}

if __name__ == "__main__":
    print("parameter grid:\n{}".format(param_grid))


grid_search, logreg_results = grid_search_wrapper(X_train, y_train, pipe, param_grid, 'roc_auc')


logreg_pipe = grid_search.best_estimator_  # The full pipeline
logreg_pipe.named_steps['logreg']  # The LogisticRegression model


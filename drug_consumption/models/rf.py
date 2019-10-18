from drug_consumption.prepare_data import X_train, y_train, X_test, y_test
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from drug_consumption.models.wrapper import grid_search_wrapper



# Here we'll fit a Random Forest Classifier. 
# This is the average of a number of decision tree classifiers with some randomness injected into the tree-building process.
# The parameters that will be searched accross are max_depth, and max_features

# max_depth limits the number of decisions that can happen in a tree; set lower to avoid overfitting
# max_features is the number of features randomly selected to fit each tree; set lower to include more randomness and avoid overfitting
#   usually set to sqrt(n_features)


ct = ColumnTransformer([
     ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse=False), ['Gender', 'Education', 'Country', 'Ethnicity'])], 
     remainder='passthrough')

pipe = Pipeline([
    ("preparation", ct),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=2))
])

param_grid = {
    'rf__max_depth': [5, 15, 25, 35, 45],
    'rf__max_features': [4, 7, 10]
}
# Looks like increasing beyong max_depth=35 doesn't improve the model

grid_search, rf_results = grid_search_wrapper(X_train, y_train, pipe, param_grid, refit_score='roc_auc')
# The best params are the same when using accuracy score or roc auc

# Store classifier for later comparison with others
rf_pipe = grid_search.best_estimator_



# Plot features importances according to Random Forest
# Having trouble with this because the column transformer doesnt name columns

# feature_importances = pd.DataFrame(rf.feature_importances_, index=canna_features.columns, columns=['importance']).sort_values('importance', ascending=False)

# def plot_rf_feature_importance():
#     n_features = canna_features.shape[1]
#     plt.barh(np.arange(n_features), rf.feature_importances_, align='center')



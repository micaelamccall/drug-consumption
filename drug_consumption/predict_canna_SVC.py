from drug_consumption.prepare_data import X_train_trans, y_train
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt


# Here, the GridSearchCV is optimized with the ROC AUC score
def grid_search_wrapper(param_grid, scorers, refit_score):
    """
    Fits a GridSearchCV classifier that optimizes with refit_score and scores with scorers
    
    Arguments: 
    param_grid = a dict with keys as strings of the parameter names and values are a list of possible values for the parameter
    scorers = a dict with keys as strings of the scorer and values that are sklearn.metrics.make_scorer(score type) objects
    refit_score = string naming the score to refit by

    Returns: tuple of GridSearchCV (for using for predictions) and results pandas DataFrame for evaluating the model
    """
    # Create a GridSearchCV instance optimized for refit_score 
    grid_search = GridSearchCV(SVC(), param_grid, scoring=scorers, refit=refit_score, cv=5, return_train_score=True, n_jobs=-1)

    # Fit on training data
    grid_search.fit(X_train_trans,y_train)

    # Store best kernel
    best_kernel = grid_search.best_params_['kernel']

    # Store results of each cross val of the best kernel
    results = pd.DataFrame(grid_search.cv_results_)
    results = results[results.param_kernel==best_kernel].loc[:,['mean_test_precision_score', 'mean_test_roc_auc', 'mean_test_accuracy', 'param_gamma', 'param_C']].round(3).reindex()
    
    if __name__ == "__main__":
        #Print best params for the chosen score and the best cross-val score
        print(
            "Parmaeters when refit for {}".format(refit_score), 
            "\n {}".format(grid_search.best_params_), 
            "\nBest cross-val roc_auc score: {}".format(np.max(results.mean_test_roc_auc)),
            "\nBest cross-val accuracy score: {:.2f}".format(np.max(results.mean_test_accuracy)))
        
      
        # Store the name of the score to plot on a heat map
        plot_score = 'mean_test_' + str(refit_score)

        # Store average cross-val score of each combo of parameters in a 6X6 array
        scores = np.array(results[plot_score]).reshape(6,6)

        # Plot the score of each combo of parameters on a heatmap
        heatmap=sns.heatmap(scores, annot=True)
        plt.title('Cross-val test scores for ' + str(refit_score))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.xticks(np.arange(6), results.param_gamma)
        plt.yticks(np.arange(6), results.param_C)
        plt.show()

    # Return both the grid search object and the results DataFrame
    returns = (grid_search, results)

    return returns

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
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}, 
    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='roc_auc')
# This seems like a better parameter grid space because the plot is filled up with evenly with the higher cross-val scores
# But the best parameters are the same as the ones we found in the first grid search

svc = grid_search.best_estimator_

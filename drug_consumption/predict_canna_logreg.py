from drug_consumption.prepare_data import X_train_trans, y_train
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


# Wrapper to fit GridSearchCV optimized with different scores:
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
    # Usually good to set n_estimators (the number of trees to build) to as high as is feasible
    grid_search = GridSearchCV(LogisticRegression(), param_grid, scoring=scorers, refit=refit_score, cv=5, return_train_score=True, n_jobs=-1)

    # Fit on training data
    grid_search.fit(X_train_trans,y_train)

    # Store results of each cross val of the best kernel
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.loc[:,['mean_test_precision_score', 'mean_test_roc_auc', 'mean_test_accuracy', 'param_C', 'param_penalty']].round(3).reindex()
    
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
        scores = np.array(results[plot_score]).reshape(2,4)

        # Plot the score of each combo of parameters on a heatmap
        heatmap=sns.heatmap(scores, annot=True)
        plt.title('Cross-val test scores for ' + str(refit_score))
        plt.xlabel('C')
        plt.ylabel('penalty type')
        plt.xticks(np.arange(4), results.param_C)
        plt.yticks(np.arange(2), results.param_penalty.unique())
        plt.show()

    # Return both the grid search object and the results DataFrame
    returns = (grid_search, results)

    return returns


scorers = {
    'roc_auc' : make_scorer(roc_auc_score),
    'precision_score' : make_scorer(precision_score),
    'accuracy' : make_scorer(accuracy_score)
}

# l1 and l2 are types of regularization; they restrict the model and prevent it from overfitting by restricting the coefficients to be near zero. 
# l1 also forces some to be exactly 0 so that they can be ignorned entirely

param_grid = {
    'C': [0.001, 0.01, 1, 100],
    'penalty': ["l1", "l2"]
}


if __name__ == "__main__":
    print("parameter grid:\n{}".format(param_grid), "\n\nscorers to evaluate predictions on test set:\n{}".format(list(scorers.keys())))


grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='roc_auc')


logreg = grid_search.best_estimator_


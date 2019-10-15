from drug_consumption.prepare_data import X_train_trans, y_train
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# Here we'll fit a Random Forest Classifier. 
# This is the average of a number of decision tree classifiers with some randomness injected into the tree-building process.
# We'll choose the best parameters for the Random Forest using GridSearchCV
# The parameters that will be searched accross are max_depth, and max_features

# max_depth limits the number of decisions that can happen in a tree; set lower to avoid overfitting
# max_features is the number of features randomly selected to fit each tree; set lower to include more randomness and avoid overfitting
#   usually set to sqrt(n_features)

# Wrapper to fit GridSearchCV optimized with different scores:
def grid_search_wrapper(param_grid, scorers, refit_score):
    """
    Fits a GridSearchCV classifier that optimizes with refit_score and scores with scorers
    Arguments: Refit score to use for optimization
    Returns: tuple of GridSearchCV (for using for predictions) and results pandas DataFrame for evaluating the model
    """
    # Create a GridSearchCV instance optimized for refit_score 
    # Usually good to set n_estimators (the number of trees to build) to as high as is feasible
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=2), param_grid, scoring=scorers, refit=refit_score, cv=5, return_train_score=True, n_jobs=-1)

    # Fit on training data
    grid_search.fit(X_train_trans,y_train)

    # Store results of each cross val of the best kernel
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.loc[:,['mean_test_precision_score', 'mean_test_roc_auc', 'mean_test_accuracy', 'param_max_features', 'param_max_depth']].round(3).reindex()
    
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
        scores = np.array(results[plot_score]).reshape(3,3)

        # Plot the score of each combo of parameters on a heatmap
        heatmap=sns.heatmap(scores, annot=True)
        plt.title('Cross-val test scores for ' + str(refit_score))
        plt.xlabel('max_features')
        plt.ylabel('max_depth')
        plt.xticks(np.arange(3), results.param_max_features.unique())
        plt.yticks(np.arange(3), results.param_max_depth.unique())
        plt.show()

    # Return both the grid search object and the results DataFrame
    returns = (grid_search, results)

    return returns



# List of scorers
scorers = {
    'roc_auc' : make_scorer(roc_auc_score),
    'precision_score' : make_scorer(precision_score),
    'accuracy' : make_scorer(accuracy_score)
}


if __name__ == "__main__":
    # Grid to search over in in GridSearchCV:
    param_grid = {
        'max_depth': [5, 15, 25],
        'max_features': [4, 7, 10]
    }

    print("parameter grid:\n{}".format(param_grid), "\n\nscorers to evaluate predictions on test set:\n{}".format(list(scorers.keys())))


    grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='roc_auc')
    grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='accuracy')


# Try some larger values of max_depth since the highest score is for max_depth=25
param_grid = {
    'max_depth': [35, 55, 75],
    'max_features': [4, 7, 10]
}
# Looks like increasing beyong max_depth=35 doesn't improve the model


grid_search, results = grid_search_wrapper(param_grid, scorers, refit_score='roc_auc')

# The best params are the same when using accuracy score or roc auc

# Store classifier for later comparison with others
rf = grid_search.best_estimator_

if __name__ == "__main__":

    # Plot features importances according to Random Forest

    # Having trouble with this because the column transformer doesnt name columns

    feature_importances = pd.DataFrame(rf.feature_importances_, index=canna_features.columns, columns=['importance']).sort_values('importance', ascending=False)

    def plot_rf_feature_importance():
        n_features = canna_features.shape[1]
        plt.barh(np.arange(n_features), rf.feature_importances_, align='center')




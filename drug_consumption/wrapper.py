import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score,precision_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt




# Wrapper to fit GridSearchCV optimized with different scores:
def grid_search_wrapper(X_train, y_train, pipe, param_grid, refit_score):
    """
    Fits a GridSearchCV classifier that optimizes with refit_score and reports ROC AUC, precision score, and accuracy
    
    Arguments: 
    X_train = training features
    y_train training target
    pipe = a pipline object that include preprocessing steps and chosen classifier
    param_grid = a dict with keys as strings of the parameter names and values are a list of possible values for the parameter
    refit_score = string naming the score to refit by

    Returns: tuple of GridSearchCV (for using for predictions) and results pandas DataFrame for evaluating the model
    """
    # Create scorers
    scorers = {
        'roc_auc' : make_scorer(roc_auc_score),
        'precision_score' : make_scorer(precision_score),
        'accuracy' : make_scorer(accuracy_score)
    }

    # Create a GridSearchCV instance optimized for refit_score 
    # Usually good to set n_estimators (the number of trees to build) to as high as is feasible
    grid_search = GridSearchCV(pipe, param_grid, scoring=scorers, refit=refit_score, cv=5, return_train_score=True, n_jobs=-1)

    # Fit on training data
    grid_search.fit(X_train,y_train)

    # Store results of each cross val of the best kernel
    results = pd.DataFrame(grid_search.cv_results_)

    # Make a list of the parameters to extract from results
    subset_results=['mean_test_precision_score', 'mean_test_roc_auc', 'mean_test_accuracy']
    params = []
    dim = []

    if 'svc__kernel' in grid_search.best_params_:
        best_kernel = grid_search.best_params_['svc__kernel']
    

    if type(param_grid) == list:
        for grid in param_grid:
            if grid['svc__kernel']==[best_kernel]:
                for param in grid.keys():
                    dim.append(len(grid[param]))

                    # Save param for graphing later
                    if param not in params:
                        params.append(param)
                dim = dim[1:]
            for param in grid.keys():
                # Save param for graphing later
                if param not in params:
                    params.append(param)
            params=params[1:]
            # Add the param to the list of column names in results
            param_str='param_'+str(param)
            if param_str not in subset_results:
                subset_results.append(param_str)
        # Selected results
        results= results[results.param_svc__kernel==best_kernel].loc[:,subset_results].round(3).reindex()  
    else:
        for param in param_grid.keys():
            # Save param for graphing later
            params.append(param)
            # Save dimensions of param for graphing
            dim.append(len(param_grid[param]))
            param_str='param_'+str(param)
            if param_str not in subset_results:
                    subset_results.append(param_str)
        # Selected results
        results = results.loc[:,subset_results].round(3).reindex()  

    #Print best params for the chosen score and the best cross-val score
    print(
        "Parmaeters when refit for {}".format(refit_score), 
        "\n {}".format(grid_search.best_params_), 
        "\nBest cross-val roc_auc score: {}".format(np.max(results.mean_test_roc_auc)),
        "\nBest cross-val accuracy score: {:.2f}".format(np.max(results.mean_test_accuracy)))
    
    # Store the name of the score to plot on a heat map
    plot_score = 'mean_test_' + str(refit_score)

    # Store average cross-val score of each combo of parameters in an array the size of the parameter grid space
    scores = np.array(results[plot_score]).reshape(dim[0],dim[1])

    # Plot the score of each combo of parameters on a heatmap
    heatmap=sns.heatmap(scores, annot=True)
    plt.title('Cross-val test scores for ' + str(refit_score))
    plt.xlabel(params[1])
    plt.ylabel(params[0])
    plt.xticks(np.arange(dim[1]), results[subset_results[-1]].unique())
    plt.yticks(np.arange(dim[0]), results[subset_results[-2]].unique())
    plt.show()

    # Return both the grid search object and the results DataFrame
    returns = (grid_search, results)

    return returns


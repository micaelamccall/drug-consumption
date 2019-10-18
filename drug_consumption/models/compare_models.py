from drug_consumption.models.logreg import logreg_pipe
from drug_consumption.models.rf import rf_pipe
from drug_consumption.models.svc import svc_pipe
from drug_consumption.prepare_data import X_train, y_train, X_test, y_test

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

ct = ColumnTransformer([
    ("num", StandardScaler(), ['Upper_Age', 'Lower_Age']), 
     ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse=False), ['Gender', 'Education', 'Country', 'Ethnicity'])], 
     remainder='passthrough')


pipe = Pipeline([
    ("preparation", ct),
    ("classifier", LogisticRegression())
])


param_grid = [
    {'classifier': [LogisticRegression(solver='liblinear')], 
    'classifier__C': [0.001, 0.01, 1, 100],
    'classifier__penalty': ["l1", "l2"]},
    {'classifier': [LogisticRegression(solver='lbfgs', penalty="l2", max_iter=400)], 
    'classifier__C': [0.001, 0.01, 1, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=200, random_state=2)],
    'classifier__max_depth': [15, 25, 35],
    'classifier__max_features': [4, 7, 10]},
    {'classifier': [SVC()],
    'classifier__C': [0.1, 1, 10, 100, 1000, 10000], 
    'classifier__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
]



# For this grid search between all classifiers, I decided to refit on ROC AUC

clf_grid = GridSearchCV(pipe, param_grid, cv=5, scoring=make_scorer(roc_auc_score))
clf_grid.fit(X_train, y_train)

print("Best classifier and params: \n{}\n".format(clf_grid.best_params_))
print("Best cross-val score: {:.2f}".format(clf_grid.best_score_))

# The best estimator was SVC
print("Test set roc-auc for SVC: {:.2f}".format(roc_auc_score(y_test, clf_grid.decision_function(X_test))))
print("Test set accuracy score for SVC: {:.2f}".format(accuracy_score(y_test, clf_grid.predict(X_test))))



# Print the classification report for all three models to see if that gives any further information about which model is best
model_list = {'Logistic Regression': logreg_pipe, 
    'Random Forest': rf_pipe, 
    'SVC': svc_pipe}

for name, model in model_list.items():
    print("Metrics for {}".format(name), "\n", classification_report(y_test, model.predict(X_test)), "\n", confusion_matrix(y_test, model.predict(X_test)))


#SVC has a similar number of incorrect classifications for each class

# One aspect of looking at these evaluation metrics is that they all use the default thresholds for the decision functions, which is 0 (positive values are classified in the positive class)
# For instnace, the model with the best ROC AUC may still not perform well if the threshold is in a suboptimal place
# Because none of these models are getting an impressively high performance, I will take a look at the ROC curve to see if the threshold of the decision function is in the optimal place (i.e. high TPR and high FPR) 

# This plots the false positive rate by the true positive rate for every threshold value in the decision curve
# Shouldn't change threshold of the decision function using the test set: HOW TO INTEGRATE INTO CROSS VAL


def plot_roc():
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_pipe.predict_proba(X_test)[:,1])
    fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, svc_pipe.decision_function(X_test))
    fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, logreg_pipe.decision_function(X_test))

    thresh_rf = np.argmin(np.abs(thresholds_rf - 0.5))
    thresh_svc = np.argmin(np.abs(thresholds_svc))
    thresh_logreg = np.argmin(np.abs(thresholds_logreg))

    plt.plot(fpr_rf, tpr_rf, label="ROC Curve Random Forest")
    plt.plot(fpr_svc, tpr_svc, label="ROC Curve SVC")
    plt.plot(fpr_logreg, tpr_logreg, label="ROC Curve Logistic Reg")

    plt.ylabel('FPR (recall)')
    plt.xlabel('TPR')
    plt.plot(fpr_svc[thresh_svc], tpr_svc[thresh_svc], 'o', markersize = 10, label = "threshold zero svc")
    plt.plot(fpr_logreg[thresh_logreg], tpr_logreg[thresh_logreg], 'x', markersize = 10, label = "threshold zero logreg")
    plt.plot(fpr_rf[thresh_rf], tpr_rf[thresh_rf], '^', markersize = 10, label = "threshold default rf")

    plt.legend(loc=4)

    plt.show()

plot_roc()

# Looks like the default threshold for the decision function is in a pretty good spot
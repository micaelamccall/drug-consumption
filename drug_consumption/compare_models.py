from drug_consumption.predict_canna_logreg import logreg
from drug_consumption.predict_canna_rf import rf
from drug_consumption.predict_canna_SVC import svc
from drug_consumption.prepare_data import X_test_trans,y_test

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Because 64% of responses are in class 1, there would be a baseline accuracy of 0.64 if we predicted all responses to be class 1
# Because this data is slightly imbalanced, ROC-AUC may be a better evaluation metric than accuracy.  
# Area under the ROC curve shows the probability that a value of the positive class will have a higher score than one of the negative class according to the decision function of the model


print("Test set roc-auc for LogisticRegression: {:.2f}".format(roc_auc_score(y_test, logreg.decision_function(X_test_trans))))
print("Test set accuracy score for LogisticRegression: {:.2f}".format(accuracy_score(y_test, logreg.predict(X_test_trans))))


print("Test set roc-auc for Random Forest: {:.2f}".format(roc_auc_score(y_test, rf.predict_proba(X_test_trans)[:,1])))
print("Test set accuracy score for Random Forest: {:.2f}".format(accuracy_score(y_test,rf.predict(X_test_trans))))


print("Test set roc-auc for SVC: {:.2f}".format(roc_auc_score(y_test, svc.decision_function(X_test_trans))))
print("Test set accuracy score for SVC: {:.2f}".format(accuracy_score(y_test, svc.predict(X_test_trans))))

# All the models are pretty similar but Logistic Regression seems to be the best


# Next we'll take a look at the ROC curve to see if the threshold of the decision function is in the optimal place

# This plots the false positive rate by the true positive rate for every threshold value in the decision curve
# Shouldn't change threshold of the decision function using the test set: HOW TO INTEGRATE INTO CROSS VAL


def plot_roc():
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test_trans)[:,1])
    fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, svc.decision_function(X_test_trans))
    fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, logreg.decision_function(X_test_trans))

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

# Looks like the default threshold for the decision function is in a pretty good spot: it balances a high TPR and high FPR




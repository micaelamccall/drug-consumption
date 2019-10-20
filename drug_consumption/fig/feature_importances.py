from drug_consumption.models.rf import rf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# These are the column names from the numpy array producted from the column transformer
ct_index = [
    'Gender_Female', 'Gender_Male', 'Education_Doctorate', 
    'Eduction_age16', 'Education_age17', 'Education_age18', 
    'Education_before16', 'Education_Masters', 'Education_Professional_Diploma/Cert', 
    'Education_SomeCollege', 'Education_UniDegree', 'Country_Australia', 
    'Country_Canada', 'Country_NewZealand', 'Country_Other', 'Country_Ireland', 
    'Country_UK', 'Country_USA', 'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Mixed_BlackAsian',
    'Ethnicity_Mixed_BlackWhite', 'Ethnicity_Mixed_WhiteAsian', 'Ethnicity_Other', 
    'Ethnicity_White', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'ImpulsiveScore', 'SS',
    'Lower_Age', 'Upper_Age']

# Create a data frame of feature importances for each feature in the transformed dataset
feature_importances = pd.DataFrame(rf.feature_importances_, index=ct_index, columns=['importance']).sort_values('importance', ascending=False)

# Plot feature importnaces 
def plot_rf_feature_importance():
    """
    A function to plot feature importances as a horizontal barplot
    """
    n_features = len(ct_index)
    plt.barh(np.arange(n_features), rf.feature_importances_, align='center')
    plt.yticks(np.arange(len(ct_index)), ct_index, size= 7)
    plt.ylabel("Feature")
    plt.xlabel("Feature Importance")

plot_rf_feature_importance()
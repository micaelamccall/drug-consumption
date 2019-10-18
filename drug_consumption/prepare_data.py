from drug_consumption.clean_data import drug_df_clean
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data for machine learning

def make_single_drug_df(df_full, drug):

    """
    A function to select a single drug as the target variable, process age variable, 
    and isolate features dataset

    Arguments: 
    df_full = the full drug pandas Dataframe
    drug = the name of the drug column of interest as a string

    Returns:
    a tuple that is the single drug pandas dataframe and the features dataframe    
    """
    cols = [
        'ID','Age', 'Gender', 'Education', 'Country', 
        'Ethnicity', 'Nscore', 'Escore', 'Oscore', 
        'Ascore', 'Cscore', 'ImpulsiveScore','SS', drug]

    df = df_full.loc[:,cols]

    # We won't one-hot encode the age feature because it is an ordinal feature. 
    # We can maintain more information about the feature by creating a new feature for the lower bound of the age and the upper bound the age.
    # This strategy maintains more information about the age than a simple mean.
    df['Lower_Age'], df['Upper_Age'] = zip(*df['Age'].map(lambda x: x.split('-')))

    # Convert lower and upper age to floats
    df.Lower_Age=df.Lower_Age.astype('float')
    df.Upper_Age=df.Upper_Age.astype('float')

    # Make dataframe of features
    features=df.drop(['ID','Age', drug], axis = 1)

    returns = (df, features)

    return returns


drug = 'Cannabis'
df, features = make_single_drug_df(df_full=drug_df_clean, drug= drug)   


if __name__ == '__main__':
    print("Shape of", drug, "data:", df.shape)

    print("Counts per response:\n", {
        n: v for n, v in zip(df[drug].value_counts().index, df[drug].value_counts())
    }, "\nProportion each response:\n", {
        n: v for n, v in zip(df[drug].value_counts().index, df[drug].value_counts() / len(df))
    })

    # Look at the distribution of the numerical variables
    df.drop(columns=['ID', drug]).hist(bins=50, figsize=(20,15))
    # They are largely normally distributed


    # Value counts for the categorical variables

    col_list = ['Age', 'Gender', 'Education', 'Country', 'Education', 'Ethnicity']
    for feature in col_list:
        print("Counts per response:\n", {
                n: v for n, v in zip(df.loc[:,feature].value_counts().index, df.loc[:, feature].value_counts())
            })

    # Pairplot of numerical features
    sns.pairplot(df, height = 4, vars=[
        'Nscore', 'Escore', 'Oscore', 
        'Ascore', 'Cscore', 'ImpulsiveScore', 'SS'], )


# Split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(features, df[drug], random_state= 42)



# Because 64% of responses are in class 1, there would be a baseline accuracy of 0.64 if we predicted all responses to be class 1
# Because this data is slightly imbalanced, ROC-AUC may be a better evaluation metric than accuracy.  
# Area under the ROC curve shows the probability that a value of the positive class will have a higher score than one of the negative class according to the decision function of the model





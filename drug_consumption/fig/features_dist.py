from drug_consumption.prepare_data import drug, df, features
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
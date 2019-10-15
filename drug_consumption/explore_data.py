from drug_consumption.prepare_data import canna_df
import matplotlib.pyplot as plt
import seaborn as sns

# Look at the distribution of the continuous variables
canna_df.drop(columns=['ID', 'Cannabis']).hist(bins=50, figsize=(20,15))
# They are largely normally distributed


# Value counts for the categorical variables

col_list = ['Age', 'Gender', 'Education', 'Country', 'Education', 'Ethnicity']
for feature in col_list:
    print("Counts per response:\n", {
            n: v for n, v in zip(canna_df.loc[:,feature].value_counts().index, canna_df.loc[:, feature].value_counts())
        })



sns.pairplot(canna_df, height = 4, vars=[
    'Nscore', 'Escore', 'Oscore', 
    'Ascore', 'Cscore', 'ImpulsiveScore', 'SS'], )
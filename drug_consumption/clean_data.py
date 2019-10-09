import pandas as pd
from drug_consumption.import_data import drug_df

def binarize_categorical_variable(df, column, yescat):
    """ Turns a categorical variable in a DataFrame into a "yes" or "no" response
    arguments: 
    df = pandas DataFrame, 
    column = quoted column name
    yes = quoted *and bracketed* list of category names that you wish to turn to "yes" 
    returns: the original pandas df with new binarized variable"""
    df[column]=df[column].astype('category')
    
    category_list = [] 

    for cat in df[column].cat.categories:
        category_list.append(cat)

    repl_dict = {}

    for cat in category_list:
        for yes in yescat:
            if cat == yes:
                repl_dict[cat] = 1   
        if repl_dict.get(cat) == None:
            repl_dict[cat] = 0

    df[column] = df[column].replace(repl_dict)

    return df

def categorize(df, column, newcat):
    df[column]=df[column].astype('category')

    category_list = []

    for cat in df[column].cat.categories:
        category_list.append(cat)

    repl_dict = {}

    for i, cat in enumerate(category_list):
        repl_dict[cat] = newcat[i]

    df[column] = df[column].replace(repl_dict)

    return df



def cleanup_drug_df():

    # Binarize the drug columns

    drug_df_clean = drug_df

    for column in drug_df.columns[13:]:
        drug_df_clean = binarize_categorical_variable(drug_df_clean, column, yescat=['CL1','Cl2','CL3','CL4','CL5','CL6'])


    # Responses from individuals who said they had done an imaginary drug probably arent repliable,
    # so they are removed

    drug_df_clean = drug_df_clean[drug_df_clean.Semeron==0]


    # The NEO scores are scaled to a mean of 0 and standard deviation of 1 already, so I dont have to rescale
    # Gender, Ethnicity, Education, and Country are encoded as real numbers, however there is no ordering betwen the levels of these features, so they should be treated as discrete
    # One-hot encoding should therefore be applied to these features 
    # First, we need to change them back to thier original categories

    new_categories = [
        ['18-24','25-34','35-44','45-54','55-64','65-100'],
        ['Female','Male'],
        ['Left school before age 16', 'Left school at age 16', 'Left school at age 17', 'Left school at age 18', 'Some college or university','Professional diploma/certificate', 'University degree', 'Masters degree', 'Doctorate degree'],
        ['USA','New Zealand','Other', 'Australia', 'Republic of Ireland','Canada', 'UK'],
        ['Black', 'Asian','White','Mixed Black/White','Other','Mixed White/Asian','Mixed Black/Asian']]

    columns_to_replace = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']

    for i, column in enumerate(columns_to_replace):
        categorize(drug_df_clean, column, new_categories[i])
    
    return drug_df_clean


# Cleanup drug DataFrame
drug_df_clean=cleanup_drug_df()

# Inspect counts for each drug column
if __name__ == '__main__':
    for column in drug_df_clean.columns[13:]:
        print(drug_df_clean[column].value_counts())


# Subset dataframe for two drugs of interest
mush_df = drug_df_clean.loc[:,['ID','Age','Gender','Education','Country', 'Ethnicity', 'Nscore','Escore','Oscore','Ascore','Cscore','ImpulsiveScore','SS','Mushroom']]
canna_df = drug_df_clean.loc[:,['ID','Age','Gender','Education','Country', 'Ethnicity', 'Nscore','Escore','Oscore','Ascore','Cscore','ImpulsiveScore','SS','Cannabis']]



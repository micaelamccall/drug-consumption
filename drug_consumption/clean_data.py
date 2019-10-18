import pandas as pd
from drug_consumption.import_data import drug_df

def binarize_categorical_variable(df, column, yescat):
    """ Fuction to turns a categorical variable in a DataFrame into a 0 or 1 response

    Arguments: 
    df = pandas DataFrame, 
    column = name of variable column as string
    yes = quoted *and bracketed* list of category names that you wish to turn to 1

    Returns: the original pandas df with new binarized variable"""

    # Change column to category dtype so that we can access it with .categories method
    df[column]=df[column].astype('category')
    
    # Create list of categories in column
    category_list = [] 
    
    for cat in df[column].cat.categories:
        category_list.append(cat)

    # Create dictionary with 1s for yes categories and 0 for no categories
    repl_dict = {}

    for cat in category_list:
        for yes in yescat:
            if cat == yes:
                repl_dict[cat] = 1   
        if repl_dict.get(cat) == None:
            repl_dict[cat] = 0

    # Replace original column in DataFrame
    df[column] = df[column].replace(repl_dict)

    return df

def new_categories(df, column, newcat):
    """ Function to replace categories in a variables with new category names
    
    Arguments:
    df =  a pandas DataFrame
    column = name of variable column as string
    newcat = a list of strings that name the new category names
    
    Output: a pandas DataFrame with new column names"""

    # Change column to category dtype so that we can access it with .categories method
    df[column]=df[column].astype('category')

    # Create list of categories in this variable
    category_list = []

    for cat in df[column].cat.categories:
        category_list.append(cat)

    # Create a dictionary of new names for each old category name
    repl_dict = {}

    for i, cat in enumerate(category_list):
        repl_dict[cat] = newcat[i]

    df[column] = df[column].replace(repl_dict)

    return df


# The NEO scores are scaled to a mean of 0 and standard deviation of 1 already, so I dont have to rescale
# Gender, Ethnicity, Education, and Country are encoded as real numbers, however there is no ordering betwen the levels of these features, so they should be treated as discrete
# One-hot encoding should therefore be applied to these features 

def cleanup_drug_df(df):
    """ A function to clean up the drug dataframe 
    
    Argument: Pandas DataFame
    
    Output: Cleaned up data as pandas DataFrame"""

    # Binarize the drug columns

    drug_df_clean = df

    for column in drug_df.columns[13:]:
        drug_df_clean = binarize_categorical_variable(drug_df_clean, column, yescat=['CL1','Cl2','CL3','CL4','CL5','CL6'])


    # Responses from individuals who said they had done an imaginary drug probably arent repliable,
    # so they are removed

    drug_df_clean = drug_df_clean[drug_df_clean.Semeron==0]

    # Change category names 
    new_categories_list = [
        ['18-24','25-34','35-44','45-54','55-64','65-100'],
        ['Female','Male'],
        ['Left school before age 16', 'Left school at age 16', 'Left school at age 17', 'Left school at age 18', 'Some college or university','Professional diploma/certificate', 'University degree', 'Masters degree', 'Doctorate degree'],
        ['USA','New Zealand','Other', 'Australia', 'Republic of Ireland','Canada', 'UK'],
        ['Black', 'Asian','White','Mixed Black/White','Other','Mixed White/Asian','Mixed Black/Asian']]

    columns_to_replace = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']

    for i, column in enumerate(columns_to_replace):
        new_categories(drug_df_clean, column, new_categories_list[i])
    
    return drug_df_clean


# Cleanup drug DataFrame
drug_df_clean=cleanup_drug_df(drug_df)

# Inspect counts for each drug column
if __name__ == '__main__':
    for column in drug_df_clean.columns[13:]:
        print(drug_df_clean[column].value_counts())
    

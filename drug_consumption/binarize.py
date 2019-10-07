# Function to use for MUA dataset
def binarize_categorical_variable(df, column, yescat):
    """ Turns a categorical variable in a DataFrame into a "yes" or "no" response
    arguments: 
    df = pandas DataFrame, 
    column = quoted column name
    yes = quoted *and bracketed* list of category names that you wish to turn to "yes" 
    returns: the original pandas df with new binarized variable"""
    df[column]=df[column].astype('category')
    
    cat_list=[] 

    for cat in df[column].cat.categories:
        cat_list.append(cat)

    repl_dict = {}

    for cat in cat_list:
        for yes in yescat:
            if cat == yes:
                repl_dict[cat] = 1   
        if repl_dict.get(cat) == None:
            repl_dict[cat] = 0

    df[column] = df[column].replace(repl_dict)

    return df

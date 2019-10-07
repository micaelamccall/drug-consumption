import pandas as pd
from binarize import binarize_categorical_variable
from import_data import drug_df


# Binarize the drug columns
yescat = ['CL1','Cl2','CL3','CL4','CL5','CL6']
drug_df_bin = drug
for column in header_list[13:]:
    drug_df_bin = binarize_categorical_variable(drug_df_bin, column, yescat)

# Responses from individuals who said they had done an imaginary drug probably arent repliable,
# so they are removed
drug_df_bin = drug_df_bin[drug_df_bin.Semeron==0]

# Inspect counts for each drug column
for column in header_list[13:]:
    print(drug_df_bin[column].value_counts())


drug_df_bin.info


# The NEO scores are scaled to a mean of 0 and standard deviation of 1 already, so I dont have to rescale
# Gender, Ethnicity, Education, and Country are encoded as real numbers, however there is no ordering betwen the levels of these features, so they should be treated as discrete
# One-hot encoding should therefore be applied to these features 



mush_df = drug_df_bin.loc[:,['ID','Age','Gender','Education','Country', 'Ethnicity', 'Nscore','Escore','Oscore','Ascore','Cscore','ImpulsiveScore','SS','Mushroom']]
canna_df = drug_df_bin.loc[:,['ID','Age','Gender','Education','Country', 'Ethnicity', 'Nscore','Escore','Oscore','Ascore','Cscore','ImpulsiveScore','SS','Cannabis']]



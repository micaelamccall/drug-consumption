from drug_consumption.clean_data import canna_df
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# We won't one-hot encode the age feature because it is an ordinal feature. 
# We can maintain more information about the feature by creating a new feature for the lower bound of the age and the upper bound the age.
# This strategy maintains more information about the age than a simple mean.
canna_df['Lower_Age'], canna_df['Upper_Age'] = zip(*canna_df['Age'].map(lambda x: x.split('-')))

canna_features=canna_df.drop(['Age', 'Cannabis'], axis = 1)

#what is the default split amount?
X_train, X_test, y_train, y_test = train_test_split(canna_features, canna_df.Cannabis, random_state= 42)

ct = make_column_transformer(
    [(['Upper_Age', 'Lower_Age'], StandardScaler()), 
     (['Gender', 'Education', 'Country', 'Ethnicity'], OneHotEncoder(sparse=False))])


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

melbourne_file_path = '../Data Sets/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
describe = melbourne_data.describe()
cols = melbourne_data.columns

melbourne_data = melbourne_data.dropna(axis=0)
# Price is the column we want to predict
# By convention, the prediction target is called y
y = melbourne_data.Price

# The columns that are inputted into our model are called "features."
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# By convention, this data is called X.
X = melbourne_data[melbourne_features]

# print(X.describe())
# print(X.head())

####################################################################################################

# Define model. Specify a number for random_state to ensure same results each run
# Many machine learning models allow some randomness in model training.
# Specifying a number for random_state ensures you get the same results in each run
# This is a good practice to use any number and model quality won't depend meaningfully on exactly what value you choose
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict(X.head()))

####################################################################################################

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
# print(mean_absolute_error(y, predicted_home_prices))

####################################################################################################

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# 1- Define model
melbourne_model = DecisionTreeRegressor(random_state=1)
# 2- Fit model
melbourne_model.fit(train_X, train_y)
# 3- Get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)


# 4- Validate the model
# print(mean_absolute_error(val_y, val_predictions))

####################################################################################################

# Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions.
# Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)  # model
    model.fit(train_X, train_y)  # fit
    preds_val = model.predict(val_X)  # predict
    mae = mean_absolute_error(val_y, preds_val)  # validate
    return (mae)


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    #print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))  # 500 is the optimal number of leaves

####################################################################################################

# The random forest uses many trees and it makes a prediction by averaging the predictions of each component tree.
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, melb_preds))


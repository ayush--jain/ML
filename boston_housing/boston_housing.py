# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit


# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MDEV']
features = data.drop('MDEV', axis = 1)
    
# data desc
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

#print data.head()

#########################################################################################################

'''CALCULATE STATS'''

# Minimum price of the data
minimum_price = np.min(prices)

# Maximum price of the data
maximum_price = np.max(prices)

# Mean price of the data
mean_price = np.mean(prices)

# Median price of the data
median_price = np.median(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)


print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price) #use :, for printing comma seperated results
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)	

###############################################################################################################

'''SPLIT DATA TO TRAINING & TESTING DATA'''

from sklearn.cross_validation import train_test_split

# Shuffle and split the data into training and testing subsets
#training size = 20%
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 42)

#################################################################################################################

'''TRAINING, FITTING, PREDICTING & CAL SCORE'''

#calculate r2 score
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    score = r2_score(y_true, y_predict)
    
    return score


#calculate the best estimator using gridsearch
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # decision tree regressor object
    regressor = DecisionTreeRegressor()

    # dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[i for i in xrange(1,11)]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


###########################################################################################################
'''PREDICTING SELLING PRICES FOR SELECTED CLIENTS'''


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "\nParameter 'max_depth' is {} for the optimal model.\n".format(reg.get_params()['max_depth'])


# Produce a matrix for client data
client_data = [[5, 34, 15], # Client 1
               [4, 55, 22], # Client 2
               [8, 7, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)




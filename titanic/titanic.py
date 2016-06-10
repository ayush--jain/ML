import numpy as np
import pandas as pd

# RMS Titanic data visualization code 
from titanic_visualizations import survival_stats
from IPython.display import display
#%matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    

# def predictions_0(data):
#     """ Model with no features. Always predicts a passenger did not survive. """

#     predictions = []
#     for _, passenger in data.iterrows():
        
#         # Predict the survival of 'passenger'
#         predictions.append(0)
    
#     # Return our predictions
#     return pd.Series(predictions)

# # Make the predictions
# predictions = predictions_0(data)

# def predictions_1(data):
#     """ Model with one feature: 
#             - Predict a passenger survived if they are female. """
    
#     predictions = []
#     for _, passenger in data.iterrows():
        
#         if (passenger['Sex'] == 'male'):
#             predictions.append(0)
#         else:
#             predictions.append(1)
    
#     # Return our predictions
#     return pd.Series(predictions)

# # Make the predictions
# predictions = predictions_1(data)

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if ((passenger['Sex'] == 'female')) and ((passenger['Pclass'] > 2)):
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)

print accuracy_score(outcomes, predictions)

#survival_stats(data, outcomes, 'Sex')
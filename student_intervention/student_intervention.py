# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")

############################################################################
'''BASIC METADATA TO UNDERSTAND DATA'''

# df =  pd.DataFrame(student_data)

# # Calculate number of students
# n_students = len(df.index)

# # Calculate number of features
# n_features = len(df.columns)-1

# # Calculate passing students
# n_passed = len(df[df['passed']== 'yes'])

# # Calculate failing students
# n_failed = len(df[df['passed']== 'no'])

# # Calculate graduation rate
# grad_rate = float(n_passed)/(n_passed + n_failed)
# grad_rate *= 100

# # Print the results
# print "Total number of students: {}".format(n_students)
# print "Number of features: {}".format(n_features)
# print "Number of students who passed: {}".format(n_passed)
# print "Number of students who failed: {}".format(n_failed)
# print "Graduation rate of the class: {:.2f}%".format(grad_rate)



#########################################################################
'''PREPROCESSING DATA FOR EASIER OPERATIONS'''

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

#Separate features
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 


# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

#preprocess X_all
X_all = preprocess_features(X_all)

############################################################
'''TRAINING TESTING DATA SPLIT'''

from sklearn import cross_validation
# Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test  = cross_validation.train_test_split(X_all, y_all, test_size = 0.24)


###############################################################################################
'''TRAINING AND PREDICTING FUNCTIONS'''

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


#################################################################################
'''CALCULATE MODEL PERFORMANCES'''
'''MODEL IMPLEMENTATIONS'''

# from sklearn import model_A
from sklearn import tree
# from skearln import model_B
from sklearn.naive_bayes import GaussianNB
#from sklearn import model_C
from sklearn import svm

# Initialize the three models
clf_A = tree.DecisionTreeClassifier()
clf_B = GaussianNB()
clf_C = svm.SVC()

# Set up the training set sizes
X_train_100 = 100
y_train_100 = 100

X_train_200 = 200
y_train_200 = 200

X_train_300 = 300
y_train_300 = 300

# Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_C, X_train.iloc[:X_train_300], y_train.iloc[:y_train_300], X_test, y_test)


####################################################################
'''OPTIMIZATION USING GRIDSEARCH'''

from sklearn import grid_search
from sklearn.metrics import make_scorer
# Create the parameters list you wish to tune
parameters = {'gamma':[0.1, 0.5, 1, 10, 100], 'C':[1, 5, 10, 100, 1000]}

# Initialize the classifier
svr = svm.SVC()

def score_func(y_test, pred):
    return f1_score(y_test, pred, pos_label='yes')

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(score_func)

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(svr, parameters, scoring=f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
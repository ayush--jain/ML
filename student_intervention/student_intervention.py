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
X_train, X_test, y_train, y_test  = cross_validation.train_test_split(X_all, y_all, test_size = 0.24, random_state = 42, stratify = y_all)


###############################################################################################
'''TRAINING AND PREDICTING FUNCTIONS'''

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    #return training time
    return (end-start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    #returns prediction time and f1 score
    return ((end-start), f1_score(target.values, y_pred, pos_label='yes'))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Train the classifier
    t1 = train_classifier(clf, X_train, y_train)
    
    # results of prediction for training
    (train_time, train_score) = predict_labels(clf, X_train, y_train)

    #results of prediction for testing
    (test_time, test_score) = predict_labels(clf, X_test, y_test)

    #return (tranining time, training score, prediction time, testing score)
    return (t1, train_score, test_time, test_score)



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
clf_A = tree.DecisionTreeClassifier(random_state=42)
clf_B = GaussianNB()
clf_C = svm.SVC()

#testing for sizes 100, 200, 300
models = [clf_A, clf_B, clf_C]
sizes = [100,200,300]

for clf in models:
    # placeholders for the training results
    num_train=[]
    time_train=[]
    time_test=[]
    f1_train=[]
    f1_test=[]

    # print the name of each estimator
    print clf.__class__.__name__

    # loop through training sizes and append results to placeholders
    for n in sizes:
        num_train.append(n)

        # TODO: get results for training size n
        (a,b,c,d) = train_predict(clf, X_train.iloc[:n], y_train[:n], X_test, y_test)

        # TODO: append results to placeholders
        time_train.append(a)
        time_test.append(c)
        f1_train.append(b)
        f1_test.append(d) 

    results = {'Training Size':sizes,'Training Time':time_train, 'Prediction Time':time_test, 'F1 Train':f1_train, 'F1 Test':f1_test}

    df_table = pd.DataFrame(data = results, columns = ['Training Size','Training Time', 'Prediction Time','F1 Train','F1 Test'])

    print df_table, "\n"



####################################################################
'''OPTIMIZATION USING GRIDSEARCH'''

from sklearn import grid_search
from sklearn.metrics import make_scorer
# Create the parameters list you wish to tune
parameters = {'gamma':[0.1, 0.5, 1, 10, 100], 'C':[1, 5, 10, 100, 1000]}

# Initialize the classifier
svr = svm.SVC()

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(svr, parameters, scoring=f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
#f1 score for training data
(a,b) = predict_labels(clf, X_train, y_train)
print "Tuned model has a training F1 score of {:.4f}.".format(b)

#f1 score for training data
(a,b) = predict_labels(clf, X_test, y_test)
print "Tuned model has a testing F1 score of {:.4f}.".format(b)
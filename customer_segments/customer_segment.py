
'''IMPORT AND LOAD DATA'''

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
import matplotlib.pyplot as plt

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    #drop region and channel columns as they are redundant to our analysis
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape) #*data.shape equiv to data.shape[0] & [1]
except:
    print "Dataset could not be loaded."

#data.describe()

##################################################################################################################
'''selecting samples for testing/analaysis purposes'''

# Three indices selected randomly to sample from the dataset
indices = [100,200,300]

# DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "\nChosen samples of wholesale customers dataset:"
print samples
#calculate mean
mean_value = np.mean(samples)

#value of samples with reference to mean value
print "\nChosen samples around mean value:"
print (samples - np.around(mean_value))

#OBSERVATION
#0th index: since sale of fresh, frozen, detergents_paper, delicatessen is much higher than the average values of them, it is likely to be a supermarket that sells various products of diff categories.
#1st index: judging by the high grocery sale relative to the average of grocery sales, it should be a grocery store.
#2nd index: judging by the high fresh product sale as compared to the average fresh product sale, it should be a fruit and vegetable vendor.

###################################################################################################

'''TRAINING, MAKING A PREDICTIVE MODEL (using a decision tree regressor) AND CALCULATING ACCURACY (in r^2) OF MODEL
   This is done to understand importance of a feature by removing it and attempting to predict it using the other features. We have taken 'Milk' here.
   (since we randomize the result to prevent bias, we calculate scores for 1000 different randomizations and calculate mean to prevent discrepancy in score)'''
						
								
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import r2_score

scores=[]
for r_state in range(1,1000):
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = pd.DataFrame(data=data).drop('Milk',1)

    # Split the data into training and testing sets using the given feature as the target
    #test size = 25%
    X_train, X_test, y_train, y_test = train_test_split(new_data, data['Milk'], test_size=0.25, random_state=r_state)

    # decision tree regressor and fitting it to the training set
    regressor = DecisionTreeRegressor(random_state=r_state)
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)

    # score of the prediction using the testing set
    score = r2_score(pred, y_test)

    scores.append(score)
    
#calculate mean of all 1000 scores
score = np.mean(scores)
print "\nR^2 score for predicting Milk is: ", score

#OBSERVATION
#A low r^2 value indicates that it cannot be predicted with too much accuracy using all the feautres we have. However since there is a positive value, there must be some features which can predict its value to a higher accuracy and hence it fits the data. So we should keep this feature for identifying customer habits.

##################################################################################################################

''' VISUALIZATION OF FEATURE DATA'''

#viualize data with diagnol showing data distribution
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
#plt.show()

'''FEATURE SCALING USING LOG'''

# Scale the data using the natural logarithm
log_data = np.log(data)

# Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()

print "\nScaled sampled data:\n"
print log_samples

###################################################################################################################

'''OUTLIER DETECTION AND REMOVAL 
   remove data points outside IQR(outside range of 1.5*(q3 - q1) from q1 or q3')'''

   #create an numpy array for storing outliers
#use numpy arrays because they are more efficient than standard python lists
all_outliers= np.array([], dtype='int64')

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3-Q1)
    
    # Display the outliers
    print "\nData points considered outliers for the feature '{}':".format(feature)
    outlier = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    print outlier
    
    #Add outlier index to all_outlier numpy array
    all_outliers = np.append(all_outliers, outlier.index.values.astype('int64'))
    
# OPTIONAL: Select the indices for data points to remove
outliers  = []

###################################
'''used for finding number of repeated values in most efficient way'''
#find out the unique values in outlier array
all_outliers, temp = np.unique(all_outliers, return_inverse=True)

#count the number of occurence of each index
counts = np.bincount(temp) #bincount must be used with np.unique to remove ambiguity

#print outliers for repeated index
for i in range(2, data.shape[1]):
    print "Present in {} features: ".format(i), all_outliers[counts==i]

#OBSERVATION
#These points should not be removed as the the values corresponding to it are outliers for only a few features but might be of relevance for the other features as they are not outliers for them.

###################################

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)



###########################################################################################################################


'''PCA analysis'''

from sklearn.decomposition import PCA
# Apply PCA to the good data with the same number of dimensions as features
pca = PCA(n_components=good_data.shape[1])
pca = pca.fit(good_data)

# Apply a PCA transformation to the sample log-data
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca)
plt.show()

#calculate cummulative variance
print "\nExplained variance cummulative"
print pca_results['Explained Variance'].cumsum()

#############################################################################################################################

'''DIMENSIONALITY REDUCTION BASED ON PCA ANALYSIS '''

#The first principal component has a total variance of: 0.4424 and the second has: 0.2766+0.4424 = 0.7190
#So we take 2 dimensions only

# Fit PCA to the good data using only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# Apply a PCA transformation the good data
reduced_data = pca.transform(good_data)

# Apply a PCA transformation to the sample log-data
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

################################################################################################################################

''' K MEANS CLUSTERING TO MAKE CLASSIFICATION AMONG CONSUMERS'''

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

max=0
'''checking clusters of all sizes b/w 2-20'''
for i in (2,20):
    # Apply K Means and fit the reduced data
    clusterer = KMeans(n_clusters=i)
    clusterer.fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    
    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    
    #check if the score is the maximum
    if score>max:
        max=score
        num_cluster=i
        

'''Fit the clusterer with the max score'''
#Apply K Means
clusterer = KMeans(n_clusters=num_cluster)
clusterer.fit(reduced_data)

# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# Find the cluster centers
centers = clusterer.cluster_centers_

# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

print "\nFor {} clusters, maximum score of {} is obtained".format(num_cluster, max)

'''visualizing the clusters'''
rs.cluster_results(reduced_data, preds, centers, pca_samples)
plt.show()

###################################################################################################################

'''DATA RECOVERY'''

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
print "\n", true_centers

#find mean of each feature and plot a pandas bar graph to visualize
true_centers = true_centers.append(data.describe().ix['50%'])
true_centers.plot(kind = 'bar', figsize = (16, 4))
plt.show()

#########################################################################################################################


''' FOR CHANGING DELIVERY SCHEDULES'''

'''We can run 2 different A/B tests on each of the clusters. 
For each of the clusters, we can divide the customers in control and experiment groups and \
try the existing schedule for the control group and the new 3 day schedule on the expirement group.

If customers from one of the clusters respond positively to the change, we can implement the new schedule \
for that particuar cluster and if customers from both the clusters respond positively, we can implement the new schedule for the entire company.'''
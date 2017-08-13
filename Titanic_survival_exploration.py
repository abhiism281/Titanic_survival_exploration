## Project 0: Titanic Survival Exploration
#In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank,
#resulting in the deaths of most of its passengers and crew.

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm

# RMS Titanic data visualization code
#from titanic_visualizations import survival_stats
from IPython.display import display
#matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)
# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)
# Show the new dataset with 'Survived' removed
display(data.head())

#Defining accuracy score function
def accuracy_score(truth, pred):
    #Returns accuracy score for input truth and predictions.
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%".format((truth == pred).mean()*100)
    else:
        #print len(truth), ",", len(pred)
        return "Number of predictions does not match number of outcomes!"
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print accuracy_score(outcomes[:5], predictions)

#Calculating accuracy assuming no one survived in the disaster
def predictions_0(data):
    #Model with no features. Always predicts a passenger did not survive.
    predictions = []
    for _, passenger in data.iterrows():
        # Predict the survival of 'passenger'
        predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_0(data)
#Score for the above model
print accuracy_score(outcomes, predictions)


#Calculating accuracy assuming only females survived
def predictions_1(data):
    #Model with one feature:
    #Predict a passenger survived if they are female.
    count = 0
    predictions = []
    for _, passenger in data.iterrows():
        #print data['Sex']
        if data['Sex'][count] == 'male':
            predictions.append(0)
        else:
            predictions.append(1)
        count+=1
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_1(data)
print accuracy_score(outcomes, predictions)

#Calculating accuracy assuming females and males aged less than 10 years survived
#in the disaster
def predictions_2(data):
    #Model with two features:
    #Predict a passenger survived if they are female.
    #Predict a passenger survived if they are male and younger than 10.
    count = 0
    predictions = []
    for _, passenger in data.iterrows():
        #print data['Age']
        if data['Sex'][count] == 'male' and float(data['Age'][count]) > 10.0:
            predictions.append(0)
        else:
            predictions.append(1)
        count += 1
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_2(data)
print accuracy_score(outcomes, predictions)

def accuracy_score_NG(truth, pred):
    #Returns accuracy score for input truth and predictions.
    # Ensure that the number of predictions matches number of outcomes
    count = 0
    if len(truth) == len(pred):
        
        # Calculate and return the accuracy as a percent
        for i in range(len(truth)):
            if truth.values[i] == pred.values[i]:
                count += 1
        return "Predictions have an accuracy of", float(count/len(truth))
    else:
        #print len(truth), ",", len(pred)
        return "Number of predictions does not match number of outcomes!"

#Preprocess the data
data_NG = full_data.fillna(0)
outcomes = data_NG['Survived']
data_NG = data_NG.drop('Survived', axis = 1)
data_NG = data_NG.drop('PassengerId', axis = 1)
data_NG = data_NG.drop('Name', axis = 1)
data_NG = data_NG.drop('Ticket', axis = 1)

#Encoding the string labels with float
le = preprocessing.LabelEncoder()
le.fit(data_NG['Sex'])
data_NG['Sex'] = le.transform(data_NG['Sex'])
le.fit(data_NG['Embarked'])
data_NG['Embarked'] = le.transform(data_NG['Embarked'])
le.fit(data_NG['Cabin'])
data_NG['Cabin'] = le.transform(data_NG['Cabin'])

def predictions_3(data):
    #display a few data
    display(data.head(10))
    
    #Model with all features using SVM:
    clf = svm.SVC()
    predictions = clf.fit(data, outcomes).predict(data)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data_NG)
print accuracy_score(outcomes, predictions)



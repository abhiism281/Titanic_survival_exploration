## Project 0: Titanic Survival Exploration
#In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank,
#resulting in the deaths of most of its passengers and crew.

import numpy as np
import pandas as pd

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
    predictions = []
    for _, passenger in data.iterrows():
        #print data['Age']
        if data['Sex'][1] == 'male' and float(data['Age'][1]) > 10.0:
            predictions.append(0)
        else:
            predictions.append(1)
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_2(data)
print accuracy_score(outcomes, predictions)



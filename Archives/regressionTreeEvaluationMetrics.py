from decisiontree import DecisionTreeRegressor as DecesionTree_scratch
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def run_DT(training_data:pd.DataFrame,scratch=True):
    if scratch == True:
        dt = DecesionTree_scratch(max_depth=4,rss_loss=True)
    else: 
        dt = DecisionTreeRegressor(max_depth=4)

    Y = training_data["ClaimNb"]
    X = training_data.drop(columns="ClaimNb")
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    dt.fit(X_train,Y_train)
    predictions = dt.predict(X_test)
    return predictions, Y_test

def r_squared(predictions,true_label):
    predictions_np = np.array(predictions)
    true_label_np = np.array(true_label)
    TSS = (true_label_np - np.mean(true_label_np))**2
    RSS = (predictions_np - true_label_np)**2
    return (TSS - RSS)/TSS

#Defining relative paths
scaled_train = r"../../data/claims_train_scaled.csv"
unscaled_train = r"../../data/claims_train_final.csv"
scaled_test = r"../../data/claims_test_final.csv"


        

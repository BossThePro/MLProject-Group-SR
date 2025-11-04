# This will contain the primary decision tree definition
import numpy as np
import pandas as pd
def decision_tree_regressor(none):
    pass

def rss(data_points:list):
    """ For a given list of data points in a region, 
        this function returns the residual sum of squares for that region"""
    RSS = 0
    region_mean = np.mean(data_points)
    for i in range(len(data_points)):
        RSS += (data_points[i] - region_mean)**2 #For each region our prediction is the mean of points in that R
    return RSS

def best_split(X:pd.DataFrame,Y:pd.DataFrame):
    """For given training data, all in the same region, this function
        returns the best feature to split on as well as the best threshold, 
        i.e one that minimizes RSS at that point"""
    
    n_samples,n_features = X.shape
    
    best_rss = float('inf')

    for i in range(n_features):  #loop over all features 
        feature = X.columns[i]
        splits = np.unique(X[feature]) # for each feature consider all possible splits
        

        for j in range(len(splits)-1): #loops over all possible splits in that feature
            split = (splits[j] + splits[j+1]) / 2
            
            #split the data on both X and Y 
            left_region = X[feature] < split
            right_region = X[feature] >= split
            
            left_labels = y[left_region]
            right_labels = y[right_region]
            
            #calculate the RSS and add it for both regions
            current_rss = rss(list(left_labels)) + rss(list(right_labels))
            
            #if we have found a RSS less than previous best then we have a new lowest RSS
            if current_rss < best_rss:
                best_rss = current_rss
                best = [feature,split]
    
    return best

def build_tree(none):
    pass

if __name__ == "__main__":
    pass

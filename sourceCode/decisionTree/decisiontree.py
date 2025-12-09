# This will contain the primary decision tree definition
import numpy as np
import pandas as pd
import math

class Node():
    def __init__(self,feature=None,split_val=None,left_child=None,right_child=None,value=None):
        self.feature = feature
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

class DecisionTreeRegressor():
    
    def __init__(self,max_depth=None,max_leaf_samples=None,min_sample_split = 2,max_leaf=None,rss_loss = True):
        self.max_depth = max_depth
        self.max_leaf_samples = max_leaf_samples #not implemented yet
        self.min_sample_split = min_sample_split
        self.max_leaf = max_leaf #maximum number of leafs in the tree
        self.n_leaf = 0
        self.root = None
        self.rss_loss = rss_loss

    def rss(self,data_points):
        """ For a given list of data points in a region, 
            this function returns the residual sum of squares for that region"""
        mean = np.mean(data_points)
        RSS = np.sum((data_points - mean)**2) #For each region our prediction is the mean of points in that R
        return RSS
    
    def mean_poisson(self,data_points):
        mean = np.mean(data_points)

        if mean == 0:
            return 0
        
        log_term = np.zeros_like(data_points, dtype=float)
        non_zeros = data_points > 0

        log_term[non_zeros] = data_points[non_zeros] * np.log(data_points[non_zeros] / mean)
        deviance = 2 * np.sum(log_term + mean - data_points)

        return deviance


    def best_split(self,x:pd.DataFrame,y:pd.DataFrame):
        """For given training data, all in the same region, this function
            returns the best feature to split on as well as the best threshold, 
            i.e one that minimizes RSS at that point"""

        n_samples,n_features = x.shape
        
        if n_samples <= 1:
            return [None,None,None]
        
        X = np.array(x)
        Y = np.array(y)
        
        best_rss = float('inf')
        best = [None,None,best_rss]
        for i in range(n_features):  #loop over all features 
            feature = x.columns[i]
            splits = np.unique(X[:,i]) # for each feature consider all possible splits
            
            for j in range(len(splits)-1): #loops over all possible splits in that feature
                split = (splits[j] + splits[j+1]) / 2
                
                #split the data on both X and Y 
                left_region = X[:,i] < split
                right_region = X[:,i] >= split
                
                left_labels = Y[left_region]
                right_labels = Y[right_region]
                
                #calculate the loss and add it for both regions
                if self.rss_loss == True:
                    current_loss = self.rss(left_labels) + self.rss(right_labels)
                else:
                    current_loss = self.mean_poisson(left_labels) + self.mean_poisson(right_labels)
                
                #if we have found a RSS less than previous best then we have a new lowest RSS
                if current_loss < best_rss:
                    best_loss = current_loss
                    best = [feature,split,best_loss]
        
        return best

    def build_tree(self,x,y,depth=0):
        """ This function recursively splits the tree using the best split function above,
            Until a stopping condition is reached, then it returns a leaf"""
        
        #stopping conditions
        if (self.max_depth != None and depth >= self.max_depth) or \
            (self.max_leaf != None and self.n_leaf >= self.max_leaf):
            self.n_leaf += 1
            return Node(value=np.mean(y))
        
        #given our training data find the best split 
        best_feature, split, rss = self.best_split(x,y)

        if best_feature == None:
            self.n_leaf += 1
            return Node(value=np.mean(y))
        
        #we need a way to save this split
        left_region = x[best_feature] < split
        right_region = x[best_feature] >= split
        
        #recursion
        l_child = self.build_tree(x[left_region],y[left_region],depth+1)
        r_child = self.build_tree(x[right_region],y[right_region],depth+1)
        
        return Node(feature=best_feature,split_val=split,left_child=l_child,right_child=r_child)
    
    def fit(self, x, y):
        self.root = self.build_tree(x, y)
    
    def traverse(self,x,node:Node):
        "Traverse the tree, to make a prediction for a given x value"
        #if there is a leaf value on the node, then return that value as the prediction
        if node.value != None:
            return(node.value)
        
        if x[node.feature] <= node.split_val:
            return self.traverse(x,node.left_child)
        else:
            return self.traverse(x,node.right_child)
    
    def predict(self, X):
        """Predict class labels for samples in an array X"""
        predictions = []
        for i in range(len(X)):
            row = X.iloc[i] 
            predictions.append(self.traverse(row, self.root))
        return predictions
        


if __name__ == "__main__":
    pass

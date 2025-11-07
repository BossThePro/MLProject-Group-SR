# This will contain the primary decision tree definition
import numpy as np
import pandas as pd
class Node():
    def __init__(self,feature=None,split_val=None,left_child=None,right_child=None,value=None):
        self.feature = feature
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

class DecesionTreeRegressor():
    
    def __init__(self,max_depth,max_leaf_samples,min_sample_split,max_leaf):
        self.max_depth = max_depth
        self.max_leaf_samples = max_leaf_samples
        self.min_sample_split = min_sample_split
        self.max_leaf = max_leaf
        self.root = None

    def rss(self,data_points):
        """ For a given list of data points in a region, 
            this function returns the residual sum of squares for that region"""
        RSS = np.sum((data_points - np.mean(data_points))**2) #For each region our prediction is the mean of points in that R
        return RSS


    def best_split(self,X:pd.DataFrame,Y:pd.DataFrame):
        """For given training data, all in the same region, this function
            returns the best feature to split on as well as the best threshold, 
            i.e one that minimizes RSS at that point"""
        
        n_samples,n_features = X.shape

        if n_samples <= 1: 
            return [None,None,None]
        
        best_rss = float('inf')
        best = [None,None,best_rss]
        for i in range(n_features):  #loop over all features 
            feature = X.columns[i]
            splits = np.unique(X[feature]) # for each feature consider all possible splits
            

            for j in range(len(splits)-1): #loops over all possible splits in that feature
                split = (splits[j] + splits[j+1]) / 2
                
                #split the data on both X and Y 
                left_region = X[feature] < split
                right_region = X[feature] >= split
                
                left_labels = Y[left_region]
                right_labels = Y[right_region]
                
                #calculate the RSS and add it for both regions
                current_rss = self.rss(list(left_labels)) + self.rss(list(right_labels))
                
                #if we have found a RSS less than previous best then we have a new lowest RSS
                if current_rss < best_rss:
                    best_rss = current_rss
                    best = [feature,split,best_rss]
        
        return best

    def build_tree(self,x,y,depth=0):
        #We need to recursively build a tree with some stopping conditions
        
        #given our training data find the best split 
        best_feature, split, rss = self.best_split(x,y)
        
        if best_feature == None:
            return Node(value=np.mean(y))
        
        if depth >= self.max_depth:
            return Node(value=np.mean(y))
        
        #we need a way to save this split
        left_region = x[best_feature] < split
        right_region = x[best_feature] >= split
        
        #recursion
        l_child = self.build_tree(x[left_region],y[left_region],depth+1)
        r_child = self.build_tree(x[right_region],y[right_region],depth+1)
        
        return Node(feature=best_feature,split_val=split,left_child=l_child,right_child=r_child)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
    
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
        predictions = np.array()
        for x in X:
            predictions.append(self.traverse(x,self.root))
        return predictions
        


if __name__ == "__main__":
    pass

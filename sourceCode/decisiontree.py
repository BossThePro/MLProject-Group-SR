import numpy as np
# This will follow the structure of the exercise session (week 8), where we define the gini impurity/entropy/information gain first and define the decision tree afterwards
# I will also attempt to follow the NumPy docstring style found here: https://numpydoc.readthedocs.io/en/latest/format.html
# Furthermore, I will be using camel case for variables and snake case for functions -> See: https://developer.mozilla.org/en-US/docs/Glossary/Camel_case https://developer.mozilla.org/en-US/docs/Glossary/Snake_case

# To compute gini, we need to use the following formula: 1 - sum(p_i^2)
# p_i is the proportion of class i 
def gini_impurity(nodeLabels: list):
    """
    This computes the gini impurity for a given list of class labels

    Parameters 
    ----------
    nodeLabels: list
        A list containing all the labels for samples in a single node

    Returns
    -------
    float
        The Gini impurity score within the range of 0 to (1-(1/C)) where C is the amount of classes in nodeLabels

    Examples
    --------
    >>> class1Labels = ['red', 'red', 'red', 'red', 'blue', 'blue']
    >>> giniImpurity(class1Labels)
    0.44444444...
    """
    # This counts the occurence of each class in nodeLabels, and computes the appropriate proportions for each class
    uniqueClasses, classCount = np.unique(nodeLabels, return_counts=True)
    proportions = classCount/len(nodeLabels)

    # This computes the gini impurity and returns it 
    gini = 1 - sum(proportions ** 2)
    return gini

def decision_tree(none):
    pass



testNodeLabels = ['red', 'red', 'red', 'red', 'blue', 'blue']
if __name__ == "__main__":
    giniValue = gini_impurity(testNodeLabels)
    print(f"Gini value for test labels: {giniValue}")
    
    

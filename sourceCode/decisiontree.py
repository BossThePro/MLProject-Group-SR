import numpy as np
# This will follow the structure of the exercise session (week 8), where we define the gini impurity/entropy/information gain first and define the decision tree afterwards
# I will also attempt to follow the NumPy docstring style found here: https://numpydoc.readthedocs.io/en/latest/format.html
# Furthermore, I will be using camel case for variables and snake case for functions -> See: https://developer.mozilla.org/en-US/docs/Glossary/Camel_case https://developer.mozilla.org/en-US/docs/Glossary/Snake_case

def gini_impurity(nodeLabels: list):
    """
    This computes the gini impurity for a given list of class labels
    Gini impurity is given by the formula: 1 - sum(p_i^2) where p_i is the proportion of class i

    Parameters 
    ----------
    nodeLabels: list
        A list containing all the labels for samples in a single node

    Returns
    -------
    float
        The Gini impurity score within the range of 0 (pure) to 1-(1/C) (impure) where C is the amount of unique classes in nodeLabels

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

def weighted_impurity(leftNodeLabels: list, rightNodeLabels: list, _impurityMeasure=gini_impurity):
    """
    This computes the weighted impurity for a given split of node labels
    Weighted impurity is given by the formula: (n_leftNodeLabels / n_totalNodeLabels) * leftImpurity + (n_rightNodeLabels / n_totalNodeLabels) * rightImpurity

    Parameters
    ----------
    leftNodeLabels: list
        A list containing all the labels for samples in the left split node
    rightNodeLabels: list 
        A list containing all the labels for samples in the right split node
    _impurityMeasure: function 
        A parameter defining the function to use to compute weighted impurity -> should not be touched hence the _ in front
    Returns
    -------
    float
        The weighted impurity score within the range of 0 (pure) to 1-(1/C) (impure) where C is the amount of unique classes in leftNodeLabels and rightNodeLabels

    Examples
    --------
    >>> testLeftNodeLabels = ['red', 'red', 'blue', 'blue']
    >>> testRightNodeLabels = ['red', 'red']
    >>> weighted_impurity(testLeftNodeLabels, testRightNodeLabels)
    0.33333333...
    """
    # This part gets the amount of left node labels, right node labels and total labels for computation later
    numberLeftNodeLabels = len(leftNodeLabels)
    numberRightNodeLabels = len(rightNodeLabels) 
    numberTotalNodeLabels = numberLeftNodeLabels + numberRightNodeLabels 
   
    # This part gets the weighted impurity and returns it 
    weightedImpurity = (numberLeftNodeLabels / numberTotalNodeLabels) * _impurityMeasure(leftNodeLabels) + (numberRightNodeLabels / numberTotalNodeLabels) * _impurityMeasure(rightNodeLabels)
    return weightedImpurity


def entropy(nodeLabels: list):
    """
    This computes the entropy for a given list of node labels
    Entropy is given by the formula: -sum(p_i * log2(p_i)) where p_i is the proportion of class i 

    Parameters
    ----------
    nodeLabels: list
        A list containing all the labels for samples in a single node

    Returns
    -------
    float
        The entropy score within the range of 0 (pure) to 1 (impure)

    Examples
    --------
    >>> class1Labels = ['red', 'red', 'red', 'red', 'blue', 'blue']
    >>> entropy(class1Labels)
    0.91829...
    """
    # This counts the occurence of each class in nodeLabels, and computes the appropriate proportions for each class
    uniqueClasses, classCount = np.unique(nodeLabels, return_counts=True)
    proportions = classCount/len(nodeLabels)

    # This computes the entropy and returns it 
    entropy = -sum(proportions * np.log2(proportions))
    return entropy

def decision_tree(none):
    pass



testNodeLabels = ['red', 'red', 'red', 'red', 'blue', 'blue']
testLeftNodeLabels = ['red', 'red', 'blue', 'blue']
testRightNodeLabels = ['red', 'red']
if __name__ == "__main__":
    giniValue = gini_impurity(testNodeLabels)
    print(f"Gini value for test labels: {giniValue}")
    entropyValue = entropy(testNodeLabels)
    print(f"Entropy value for test labels: {entropyValue}")
    weightedImpurity = weighted_impurity(testLeftNodeLabels, testRightNodeLabels)
    print(f"Weighted impurity value for test labels: {weightedImpurity}")
    
    

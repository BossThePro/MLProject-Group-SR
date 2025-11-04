import decisiontree as dt 

testLabels = ["Red", "Red", "Red", "Blue", "Blue"]
testLeftLabels = ["Blue", "Red"]
testRightLabels = ["Red", "Red", "Blue"]
gini = dt.gini_impurity(testLabels)
entropy = dt.entropy(testLabels)
weightedImpurity = dt.weighted_impurity(testLeftLabels, testRightLabels)

print(f"Gini impurity for testLabels: {gini}")
print(f"Entropy for testLabels: {entropy}")
print(f"Weighted Impurity for split test labels: {weightedImpurity}")


def foo():
    """
    Hello

    """

foo()

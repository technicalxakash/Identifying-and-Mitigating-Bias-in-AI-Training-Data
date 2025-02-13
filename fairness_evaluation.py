# Group Positive Rate
group_positive_rate = pd.crosstab(y_test, y_pred, normalize='index')
print("Group Positive Rates:\n", group_positive_rate)

# Grouping and visualization
print(data.groupby(['race', 'sex'])['income'].count())
sns.countplot(data=data, x='income', hue='race')
plt.title('Income Distribution by Race')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns, sep=',\s*', engine='python')

# Analyze and visualize
print(data['race'].value_counts())
sns.countplot(data=data, x='race')
plt.title('Class Distribution Before Balancing')
plt.show()

from sklearn.utils import resample

# Augment the minority class
minority = data[data['race'] == 'Black']
synthetic_data = resample(minority, replace=True, n_samples=5000, random_state=42)
balanced_data = pd.concat([data, synthetic_data])

sns.countplot(data=balanced_data, x='race')
plt.title('Class Distribution After Balancing')
plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
data_path = "D:\FairDataSampling\data\imbalanced_dataset.csv"  # Adjust the path as needed
data = pd.read_csv(data_path)

# Clean up column names
data.columns = data.columns.str.strip()

# Ensure the column 'income' exists before processing
if 'income' in data.columns:
    # Separate features and target
    X = data.drop('income', axis=1)
    y = data['income']
else:
    print("The 'income' column was not found.")
    print(f"Available columns: {data.columns}")
    exit()

# Convert categorical columns to numerical
categorical_columns = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_columns)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)

# Output the new class distribution
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Save the resampled dataset
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X_encoded.columns), pd.Series(y_resampled, name='income')], axis=1)
resampled_data.to_csv("D:\FairDataSampling\data\imbalanced_dataset.csv", index=False)
print("Resampled dataset saved to 'D:\FairDataSampling\data\imbalanced_dataset.csv'.")

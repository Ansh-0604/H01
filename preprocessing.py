import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dask.diagnostics import ProgressBar

# File paths
train_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_training-set.parquet'
test_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_testing-set.parquet'

# Load the datasets
train_df = dd.read_parquet(train_path)
test_df = dd.read_parquet(test_path)

# Inspect the datasets
print(train_df.info())
print(test_df.info())

# Identify numeric columns
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Handle missing values for numeric columns only
with ProgressBar():
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median()).persist()
    test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median()).persist()

# Encode categorical variables
label_encoders = {}
categorical_cols = train_df.select_dtypes(include=['category']).columns

with ProgressBar():
    for col in categorical_cols:
        le = LabelEncoder()

        # Fit on the combined unique values from both train and test datasets
        combined = dd.concat([train_df[col], test_df[col]], interleave_partitions=True).drop_duplicates().compute()
        le.fit(combined)

        # Transform train data
        train_df[col] = train_df[col].map_partitions(lambda x: le.transform(x), meta=('x', 'i4')).persist()

        # Transform test data
        test_df[col] = test_df[col].map_partitions(lambda x: le.transform(x), meta=('x', 'i4')).persist()
        
        label_encoders[col] = le  # Store the encoder for potential later use

# Continue with the rest of your processing...

# Separate features and target (assuming the target column is named 'label')
X_train = train_df.drop('label', axis=1).persist()
y_train = train_df['label'].persist()

X_test = test_df.drop('label', axis=1).persist()
y_test = test_df['label'].persist()

# Normalize numerical features using StandardScaler
scaler = StandardScaler()

# Compute train and test features
X_train_scaled = scaler.fit_transform(X_train.compute())
X_test_scaled = scaler.transform(X_test.compute())

# Reshape data for CNN-LSTM input format: (samples, timesteps, features)
n_timesteps = 5

# Ensure features can be evenly divided by the number of timesteps
n_features = X_train_scaled.shape[1] // n_timesteps

# Reshape the train and test data to have multiple timesteps
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], n_timesteps, n_features))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], n_timesteps, n_features))

# Save preprocessed datasets back as .parquet files
train_preprocessed_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_training-set-preprocessed.parquet'
test_preprocessed_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_testing-set-preprocessed.parquet'

# Convert the reshaped arrays back to DataFrame to save in parquet format
train_df_preprocessed = dd.from_array(X_train_reshaped.reshape(X_train_reshaped.shape[0], -1), 
                                       columns=[f'feature_{i}' for i in range(X_train_reshaped.shape[1] * X_train_reshaped.shape[2])])
test_df_preprocessed = dd.from_array(X_test_reshaped.reshape(X_test_reshaped.shape[0], -1), 
                                      columns=[f'feature_{i}' for i in range(X_test_reshaped.shape[1] * X_test_reshaped.shape[2])])

# Save the data
with ProgressBar():
    # Removed write_options and replaced with compression directly
    train_df_preprocessed.to_parquet(train_preprocessed_path, compression='snappy')
    test_df_preprocessed.to_parquet(test_preprocessed_path, compression='snappy')

print(f"Preprocessed data saved to {train_preprocessed_path} and {test_preprocessed_path}.")

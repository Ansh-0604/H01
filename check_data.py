import pandas as pd

# Load the original training data
original_data_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_training-set.parquet'
original_train_df = pd.read_parquet(original_data_path)

# Display the first few rows of the dataset
print("First few rows of the original training dataset:")
print(original_train_df.head())

# Print the column names
print("Columns in original training data:")
print(original_train_df.columns)

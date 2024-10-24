import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
output_dir = '/home/kali/H01/EDA'
os.makedirs(output_dir, exist_ok=True)

# Load preprocessed datasets
train_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_training-set-preprocessed.parquet'
test_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_testing-set-preprocessed.parquet'
train_df = dd.read_parquet(train_path)
test_df = dd.read_parquet(test_path)

# Basic overview
print("Train DataFrame Info:")
print(train_df.info())
print("\nTest DataFrame Info:")
print(test_df.info())

# Print column names
print("\nTraining set columns:", train_df.columns.tolist())
print("Testing set columns:", test_df.columns.tolist())

# Descriptive statistics
train_stats = train_df.describe().compute()
print("\nTrain Descriptive Statistics:")
print(train_stats)

test_stats = test_df.describe().compute()
print("\nTest Descriptive Statistics:")
print(test_stats)

# Identify numeric columns
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Histograms for numeric features
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    train_df[col].compute().hist(bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, f'histogram_{col}.png'))  # Save histogram
    plt.close()

# Correlation matrix
corr = train_df.corr().compute()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))  # Save correlation matrix
plt.close()

# Box plots for categorical features (Remove or update if 'label' is not found)
if numeric_cols:
    plt.figure(figsize=(10, 5))
    # Use the first numeric column for the box plot (update this as needed)
    plt.boxplot(train_df[numeric_cols[0]].compute())
    plt.title(f'Box plot of {numeric_cols[0]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel('Values')
    plt.savefig(os.path.join(output_dir, f'boxplot_feature0.png'))  # Save box plot
    plt.close()
else:
    print("No numeric features found in the DataFrame.")

# Identify missing values
missing_train = train_df.isnull().sum().compute()
missing_test = test_df.isnull().sum().compute()

print("\nMissing Values in Train Data:")
print(missing_train[missing_train > 0])

print("\nMissing Values in Test Data:")
print(missing_test[missing_test > 0])

# Feature distributions using KDE plots
plt.figure(figsize=(15, 10))
for col in numeric_cols:
    sns.kdeplot(train_df[col].compute(), label=col)
plt.title('Feature Distributions', fontsize=16)
plt.legend()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))  # Save KDE plot
plt.close()

# Target variable analysis (Commented out until target variable is confirmed)
# if 'label' in train_df.columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x='label', data=train_df.compute())  # Replace 'label' with the actual target column name
#     plt.title('Distribution of Target Variable', fontsize=16)
#     plt.savefig(os.path.join(output_dir, 'target_variable_distribution.png'))  # Save target distribution
#     plt.close()
# else:
#     print("Target variable 'label' not found in the DataFrame.")

print("Exploratory Data Analysis complete.")

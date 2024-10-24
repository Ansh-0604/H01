import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = '/home/kali/H01/UNSW-NB15/UNSW_NB15_training-set.parquet'
df = pd.read_parquet(data_path)

# Display the first few rows and column names
print("First few rows of the dataset:")
print(df.head())
print("Columns in dataset:", df.columns)

# Define features and target variable
target_variable = 'label'
feature_columns = df.columns[df.columns != target_variable]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[feature_columns], drop_first=True)

# Prepare the feature matrix and target vector
X = df_encoded.values
y = df[target_variable].values

# Introduce more noise to the input features
noise_factor = 0.3  # Increased noise
X += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build a simple Dense model instead of LSTM
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.8))  # Increased dropout to 0.8
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a lower learning rate
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for fewer epochs
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Save the model
model.save('/home/kali/H01/Saved_Model/model.keras')

print("Model saved successfully!")

# Plot training & validation accuracy and loss
plt.figure(figsize=(14, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Show plots
plt.tight_layout()
plt.show()

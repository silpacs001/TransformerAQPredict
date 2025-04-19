import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the dataset (assuming CSV format)
df = pd.read_csv('openaq.csv', sep=';')

# Display initial rows of the dataframe to understand the data
print(df.head())

# Handle missing values (you can choose to fill them with mode/mean/median or drop)
df = df.fillna(df.mode().iloc[0])  # Or use df.dropna() to remove rows with missing values

# Define categorical columns and numerical columns
categorical_cols = ['Country Code', 'City', 'Location', 'Pollutant', 'Source Name', 'Unit', 'Last Updated', 'Country Label']
numerical_cols = ['Value']

# Encode categorical features using LabelEncoder for simplicity (or you can use OneHotEncoder)
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Prepare features (X) and target (y) variables
X_data = df[categorical_cols + numerical_cols].values
y_data = df['Value'].values  # Assuming 'Value' is your target variable

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[:, -1:] = scaler.fit_transform(X_train[:, -1:])  # Apply to 'Value' column
X_test[:, -1:] = scaler.transform(X_test[:, -1:])

# Standardize the target variable (y)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))

# Build the Transformer model using the Functional API
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Reshape the input to be 3D (batch_size, sequence_length, feature_dimension)
    reshaped_input = layers.Reshape((1, input_shape[0]))(inputs)  # Adding a time dimension

    # Add a MultiHeadAttention layer
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(reshaped_input, reshaped_input)
    attention_output = layers.GlobalAveragePooling1D()(attention_output)

    # Dense layers
    x = layers.Dense(64, activation='relu')(attention_output)
    x = layers.Dense(32, activation='relu')(x)

    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Build the model
transformer_model = build_transformer_model(X_train.shape[1:])

# Compile the model
transformer_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
transformer_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss = transformer_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = transformer_model.predict(X_test)

# Convert the predictions back to the original scale
y_pred_original_scale = y_scaler.inverse_transform(y_pred)

# Print out the first 10 predictions on the original scale
print('First 10 Predictions (Original Scale):', y_pred_original_scale[:10])

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R² (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

# Visualize Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(100), y_scaler.inverse_transform(y_test[:100]), color='blue', label='Actual', alpha=0.5)
plt.scatter(np.arange(100), y_pred_original_scale[:100], color='red', label='Predicted', alpha=0.5)
plt.xlabel('Samples')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

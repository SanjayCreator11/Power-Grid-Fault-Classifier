# train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib # Library to save your model

print("Script started: Training the final model.")

# Load the dataset
df = pd.read_csv('Fault_Data_Updated.csv')

# Feature Engineering
df['Current_Diff_A'] = df['Current_Send_A'] - df['Current_Recv_A']
df['Current_Diff_B'] = df['Current_Send_B'] - df['Current_Recv_B']
df['Current_Diff_C'] = df['Current_Send_C'] - df['Current_Recv_C']

# Prepare the data using the ENTIRE dataset
X = df.drop('Fault_Type', axis=1)
y = df['Fault_Type']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the final ANN model
model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42, early_stopping=True, n_iter_no_change=20)

print("Training the ANN model on the full dataset...")
model.fit(X_scaled, y)
print("Model training complete.")

# Save the model and the scaler to files
joblib.dump(model, 'fault_classifier_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully.")
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and the scaler
model = joblib.load('fault_classifier_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set the page title and a brief description
st.set_page_config(page_title="Fault Classifier", layout="wide")
st.title("Transmission Line Fault Classifier ⚡")
st.write("This app uses a Neural Network to predict the type of fault on a transmission line based on your input.")

# --- User Input in the Sidebar ---
st.sidebar.header("Input Electrical Measurements")

def user_input_features():
    # Create input fields for all features
    resistance = st.sidebar.number_input('Resistance (R)', 0.0, 1000.0, 500.0)
    current_bus_a = st.sidebar.slider('Current Bus25 A', 0.0, 10.0, 5.1)
    current_bus_b = st.sidebar.slider('Current Bus25 B', 0.0, 10.0, 3.2)
    current_bus_c = st.sidebar.slider('Current Bus25 C', 0.0, 10.0, 2.2)
    current_recv_a = st.sidebar.slider('Current Recv A', 0.0, 0.1, 0.003)
    current_recv_b = st.sidebar.slider('Current Recv B', 0.0, 0.1, 0.003)
    current_recv_c = st.sidebar.slider('Current Recv C', 0.0, 0.1, 0.003)
    current_send_a = st.sidebar.slider('Current Send A', 0.0, 0.1, 0.003)
    current_send_b = st.sidebar.slider('Current Send B', 0.0, 0.1, 0.003)
    current_send_c = st.sidebar.slider('Current Send C', 0.0, 0.1, 0.003)
    voltage_bus_a = st.sidebar.slider('Voltage Bus25 A', 0.0, 1000.0, 842.0)
    voltage_bus_b = st.sidebar.slider('Voltage Bus25 B', 0.0, 1000.0, 667.0)
    voltage_bus_c = st.sidebar.slider('Voltage Bus25 C', 0.0, 1000.0, 334.0)
    power_gen_p = st.sidebar.number_input('Power Gen P', 0.0, 2000.0, 1040.0)
    power_gen_q = st.sidebar.number_input('Power Gen Q', 0.0, 2000.0, 1540.0)
    power_recv_p = st.sidebar.number_input('Power Recv P', 0.0, 0.001, 0.00005, format="%.5f")
    power_recv_q = st.sidebar.number_input('Power Recv Q', 0.0, 0.001, 0.00005, format="%.5f")
    power_send_p = st.sidebar.number_input('Power Send P', 0.0, 0.001, 0.00013, format="%.5f")
    power_send_q = st.sidebar.number_input('Power Send Q', 0.0, 0.001, 0.00014, format="%.5f")

    data = {'Resistance(R)': resistance, 'Current_Bus25_A': current_bus_a, 'Current_Bus25_B': current_bus_b, 'Current_Bus25_C': current_bus_c, 'Current_Recv_A': current_recv_a, 'Current_Recv_B': current_recv_b, 'Current_Recv_C': current_recv_c, 'Current_Send_A': current_send_a, 'Current_Send_B': current_send_b, 'Current_Send_C': current_send_c, 'Voltage_Bus25_A': voltage_bus_a, 'Voltage_Bus25_B': voltage_bus_b, 'Voltage_Bus25_C': voltage_bus_c, 'PowerGen_P': power_gen_p, 'PowerGen_Q': power_gen_q, 'PowerRecv_P': power_recv_p, 'PowerRecv_Q': power_recv_q, 'PowerSend_P': power_send_p, 'PowerSend_Q': power_send_q}

    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Feature Engineering on the user's input
df_input['Current_Diff_A'] = df_input['Current_Send_A'] - df_input['Current_Recv_A']
df_input['Current_Diff_B'] = df_input['Current_Send_B'] - df_input['Current_Recv_B']
df_input['Current_Diff_C'] = df_input['Current_Send_C'] - df_input['Current_Recv_C']

# Display the user's input
st.subheader("User Input Parameters")
st.write(df_input)

# Prediction Button
if st.button("Predict Fault Type"):
    # Scale the user input
    input_scaled = scaler.transform(df_input)

    # Make a prediction
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result")
    st.success(f"The model predicts Fault Type: {prediction[0]}")
import streamlit as st
import numpy as np
import cloudpickle

# Define the load_model function with a parameter (file_path) for the file name
def load_model(file_path):
    try:
        # Attempt to load with joblib
        return joblib.load(file_path)
    except:
        # If joblib fails, try loading with cloudpickle
        with open(file_path, "rb") as f:
            return cloudpickle.load(f)

# Specify the model file path
model_file = 'trained_rf_model.pkl'  # or 'trained_rf_model.joblib' if using joblib
model = load_model(model_file)

# Streamlit App Title and Description
st.title("Credit Card Fraud Detection App")
st.write("This application predicts the likelihood of fraud based on transaction details.")

# Input Form
st.header("Enter Transaction Details")

# Collect user input for each feature
time = st.number_input("Time (in seconds)", min_value=0, step=1, value=0)
amount = st.number_input("Amount (in dollars)", min_value=0.0, step=1.0, value=0.0)

# Collect user input for V1 to V28 features
v_features = []
for i in range(1, 29):
    v_features.append(st.number_input(f"V{i}", value=0.0))

# When the Predict button is clicked, make predictions
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[time, amount] + v_features])

    # Make predictions
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0, 1]  # Probability of fraud

    # Display the results
    if prediction == 1:
        st.write(f"Warning: High probability of fraud ({prediction_prob:.2f})")
    else:
        st.write(f"Low probability of fraud ({prediction_prob:.2f})")


import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load trained model
if not os.path.exists("rent_model.pkl"):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

model_data = joblib.load("rent_model.pkl")

model = model_data["model"]
cities = model_data["cities"]

# Application heading
st.title("🏠 House Rent Prediction")

st.write(
    "Predict the estimated monthly rent of a house "
    "based on its location and size."
)

st.divider()

# User input
st.subheader("Enter House Details")

city = st.selectbox(
    "Select City",
    cities
)

size = st.number_input(
    "House Size (Square Feet)",
    min_value=1.0,
    max_value=max(10000.0, model_data["max_size"]),
    value=float(min(500.0, max(1.0, model_data["max_size"]))),
    step=50.0
)

st.write("")

# Prediction button
if st.button("Predict Rent", use_container_width=True):

    input_data = pd.DataFrame({
        "City": [city],
        "Size": [size]
    })

    prediction = model.predict(input_data)[0]

    st.success("Rent Prediction Successful!")

    st.subheader("Estimated Monthly Rent")

    st.markdown(
        f"<h1 style='color:green;'>₹{prediction:,.0f}</h1>",
        unsafe_allow_html=True
    )

    st.caption(
        "This is an estimated rent based on the trained "
        "machine learning model."
    )

# Project information
st.divider()

with st.expander("About This Project"):

    st.write("Machine Learning Project: House Rent Prediction")

    st.write("Algorithm: Random Forest Regressor")

    st.write("Input Features: City and House Size")

    st.write("Output: Estimated Monthly Rent")

    st.write(
        f"Model RMSE: ₹{model_data['rmse']:,.2f}"
    )

    st.write(
        f"Dataset Records: {model_data['total_records']}"
    )

st.caption("Developed as a Machine Learning College Project.")

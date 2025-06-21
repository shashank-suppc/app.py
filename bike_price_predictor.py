import numpy as np
import pandas as pd
import streamlit as st

# Load data and train model
df = pd.read_csv('bike_changes3.csv')

# Calculate brand means
brand_means = {}
for brand in df['brand_name'].unique():
    brand_means[brand] = np.mean(df[df['brand_name'] == brand]['price'])

df['brand_name_mean'] = df['brand_name'].map(brand_means)

# Prepare features and target
X = df.iloc[:, [0, 1, 2, 4, 5, 8, 10]]
y = df.iloc[:, 6]

# Train model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Know your bike price")

# Initialize session state
if 'show_form' not in st.session_state:
    st.session_state.show_form = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False


# Prediction function
def predict_price(inputs):
    features = [
        inputs['model_year'],
        inputs['kms_driven'],
        inputs['owner'],
        inputs['mileage'],
        inputs['power'],
        inputs['CC'],
        brand_means[inputs['brand_name']]]
    return model.predict([features])[0]


# Show prediction button only if form is not shown
if not st.session_state.show_form:
    if st.button('Click here to predict'):
        st.session_state.show_form = True

# Show form when requested
if st.session_state.show_form:
    with st.form("bike_prediction_form"):
        name = st.text_input("Enter your name")
        model_year = st.number_input("Enter model year of your bike", min_value=1900, max_value=2025)
        kms_driven = st.number_input("How many kms your bike has covered yet?", min_value=0)
        owner = st.selectbox("How many times ownership has been transferred?", sorted(df['owner'].unique()))
        mileage = st.number_input("How much mileage your bike produces?", min_value=0.0)
        power = st.number_input("How much power your bike gives (in bhp)?", min_value=0.0)
        CC = st.number_input("Enter bike CC", min_value=0)
        brand_name = st.selectbox("Choose your brand:", sorted(brand_means.keys()))

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state.inputs = {
                'name': name,
                'model_year': model_year,
                'kms_driven': kms_driven,
                'owner': owner,
                'mileage': mileage,
                'power': power,
                'CC': CC,
                'brand_name': brand_name
            }
            st.session_state.submitted = True

# Show prediction after form submission
if st.session_state.submitted:
    inputs = st.session_state.inputs
    price = predict_price(inputs)
    st.success(f"Dear {inputs['name']}, the estimated price of your bike is ₹{price:,.2f}")

    # Reset button
    if st.button('Predict another bike'):
        st.session_state.show_form = False
        st.session_state.submitted = False
        st.experimental_rerun()

# Show image at the bottom
st.image("Screenshot 2025-06-21 101705.png")
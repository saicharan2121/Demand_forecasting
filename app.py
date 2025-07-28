import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime

# Load your existing model and helper files
try:
    model = joblib.load('demand_forecasting_model.joblib')
    with open('unique_ids.json', 'r') as f:
        unique_ids = json.load(f)
    # FIX: Using the correct filename from your screenshot
    with open('training_columns (1).json', 'r') as f:
        training_columns = json.load(f)
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please check your filenames and make sure they are all in the same folder as your app.")
    st.stop()


st.set_page_config(page_title="Demand Forecaster", page_icon="📈", layout="wide")
st.title('📈 Demand Forecasting Dashboard')
st.write("Enter the product and store details to forecast the number of units sold.")

# --- INPUT WIDGETS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Product & Store")
    store_id = st.selectbox('Select Store ID', options=unique_ids['store_ids'])
    sku_id = st.selectbox('Select SKU ID', options=unique_ids['sku_ids'])

with col2:
    st.header("Date & Price")
    forecast_date = st.date_input("Select Forecast Date", datetime.now())
    total_price = st.number_input('Total Price', min_value=0.0, value=150.0, format="%.2f")
    base_price = st.number_input('Base Price', min_value=0.0, value=200.0, format="%.2f")

with col3:
    st.header("Promotional Status")
    is_featured_sku = st.checkbox("Is this a featured SKU?")
    is_display_sku = st.checkbox("Is this an SKU on display?")


# --- PREDICTION LOGIC ---
if st.button('Predict Units Sold'):
    input_data = {
        'record_ID': [0],
        'total_price': [total_price],
        'base_price': [base_price],
        'is_featured_sku': [1 if is_featured_sku else 0],
        'is_display_sku': [1 if is_display_sku else 0],
        'day': [forecast_date.day],
        'month': [forecast_date.month],
        'year': [forecast_date.year],
        f'store_{store_id}': [1],
        f'item_{sku_id}': [1]
    }

    input_df = pd.DataFrame(input_data)
    final_df = input_df.reindex(columns=training_columns, fill_value=0)

    try:
        prediction = model.predict(final_df)
        predicted_units = int(round(prediction[0]))
        st.success(f"**Predicted Units Sold: {predicted_units}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.info("**Note on `record_ID`**: Your model was trained using the `record_ID` column. For better model performance in the future, it's recommended to remove identifier columns like this before training.")
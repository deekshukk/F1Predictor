import streamlit as st
import pandas as pd
import joblib
from feature import create_features_and_target

# Load model and data
model = joblib.load("model/random_forest_f1_model.pkl")
features, target = create_features_and_target()
raw_data = pd.read_csv('data/processed/f1_2025_multiple_races.csv')

st.title("🏎️ F1 2025 Finishing Position Predictor")

drivers = sorted(raw_data['Driver'].unique())
driver_name = st.selectbox("Select Driver", drivers)
grid_position = st.slider("Enter Starting Grid Position", 1, 20, 10)

if st.button("Predict Finishing Position"):

    driver_data = raw_data[raw_data['Driver'] == driver_name].sort_values(by=["Year", "Race"], ascending=False).iloc[0]

    # Fill in features (using last known data for driver)
    input_features = pd.DataFrame([{
        'GridPosition': grid_position,
        'Team_enc': driver_data['Team_enc'],
        'Driver_enc': driver_data['Driver_enc'],
        'Race_enc': driver_data['Race_enc'],
        'Race_number': driver_data['Race_number'],
        'Driver_avg_finish': driver_data['Driver_avg_finish'],
        'Driver_cum_points': driver_data['Driver_cum_points'],
        'Driver_finish_std': driver_data['Driver_finish_std'],
        'Team_avg_finish': driver_data['Team_avg_finish'],
        'Grid_vs_Finish': grid_position - driver_data['Position'],
        'Finished': 1
    }])

    # Predict and show result
    pred = model.predict(input_features)[0]
    st.success(f"🏁 Predicted Finishing Position for {driver_name}: **{pred:.2f}**")

import streamlit as st
import pandas as pd
import joblib
from feature import create_features_and_target
from sklearn.preprocessing import LabelEncoder

# === Load trained model ===
model = joblib.load("model/random_forest_f1_model.pkl")

# === Load and preprocess raw CSV ===
raw_df = pd.read_csv("data/processed/f1_2025_multiple_races.csv")
raw_df = raw_df.sort_values(by=["Year", "Race", "Driver"]).reset_index(drop=True)

# === Label Encoding (same as feature.py) ===
le_driver = LabelEncoder()
le_driver.fit(raw_df['Driver'])
raw_df['Driver_enc'] = le_driver.transform(raw_df['Driver'])

le_team = LabelEncoder()
le_team.fit(raw_df['Team'])
raw_df['Team_enc'] = le_team.transform(raw_df['Team'])

le_race = LabelEncoder()
le_race.fit(raw_df['Race'])
raw_df['Race_enc'] = le_race.transform(raw_df['Race'])

# === Race number per season ===
race_order = raw_df[['Year', 'Race', 'Race_enc']].drop_duplicates().sort_values(['Year', 'Race_enc'])
race_order['Race_number'] = range(1, len(race_order) + 1)
raw_df = raw_df.merge(race_order[['Year', 'Race', 'Race_number']], on=['Year', 'Race'], how='left')

# === Driver rolling stats (simplified defaults if not enough data) ===
raw_df['Position'] = pd.to_numeric(raw_df['Position'], errors='coerce')
raw_df['Points'] = pd.to_numeric(raw_df['Points'], errors='coerce').fillna(0)

raw_df['Driver_avg_finish'] = (
    raw_df.groupby('Driver')['Position']
          .transform(lambda x: x.shift().expanding().mean())
).fillna(raw_df['Position'].mean())

raw_df['Driver_cum_points'] = (
    raw_df.groupby('Driver')['Points']
          .transform(lambda x: x.shift().cumsum())
).fillna(0)

raw_df['Driver_finish_std'] = (
    raw_df.groupby('Driver')['Position']
          .transform(lambda x: x.shift().expanding().std())
).fillna(0)

raw_df['Team_avg_finish'] = (
    raw_df.groupby('Team')['Position']
          .transform(lambda x: x.shift().expanding().mean())
).fillna(raw_df['Position'].mean())

# === Grid vs Finish ===
raw_df['Grid_vs_Finish'] = raw_df['GridPosition'] - raw_df['Position']

# === Finished flag ===
raw_df['Finished'] = raw_df['Status'].apply(lambda x: 1 if str(x).lower() == 'finished' else 0)

# === Streamlit UI ===
st.title("🏎️ F1 2025 Predictor")

drivers = sorted(raw_df['Driver'].unique())
driver_name = st.selectbox("Select Driver", drivers)
grid_position = st.slider("Starting Grid Position", 1, 20, 10)

if st.button("Predict Finishing Position"):
    driver_row = raw_df[raw_df['Driver'] == driver_name].sort_values(by=["Year", "Race"], ascending=False).iloc[0]

    input_row = {
        'GridPosition': grid_position,
        'Team_enc': driver_row['Team_enc'],
        'Driver_enc': driver_row['Driver_enc'],
        'Race_enc': driver_row['Race_enc'],
        'Race_number': driver_row['Race_number'],
        'Driver_avg_finish': driver_row['Driver_avg_finish'],
        'Driver_cum_points': driver_row['Driver_cum_points'],
        'Driver_finish_std': driver_row['Driver_finish_std'],
        'Team_avg_finish': driver_row['Team_avg_finish'],
        'Grid_vs_Finish': grid_position - driver_row['Position'],
        'Finished': 1
    }

    input_df = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]
    st.success(f"🏁 Predicted Finishing Position for {driver_name}: **{prediction:.2f}**")

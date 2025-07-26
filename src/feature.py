import pandas as pd

# Load your CSV
df = pd.read_csv('race_data.csv')

# Encode Status to numeric
status_mapping = {
    'Finished': 0,
    'Retired': 1,
    'DNF': 2,
    # add other statuses if needed
}
df['StatusEncoded'] = df['Status'].map(status_mapping)

# Create FinishedFlag: 1 if finished, 0 otherwise
df['FinishedFlag'] = df['Status'].apply(lambda x: 1 if x == 'Finished' else 0)

# Encode Driver and Team (Label Encoding)
from sklearn.preprocessing import LabelEncoder

driver_encoder = LabelEncoder()
team_encoder = LabelEncoder()

df['DriverEncoded'] = driver_encoder.fit_transform(df['Driver'])
df['TeamEncoded'] = team_encoder.fit_transform(df['Team'])

# Create laps completion ratio if you know total race laps (assume total_laps=70 for example)
total_laps = 70
df['LapsRatio'] = df['Laps'] / total_laps

# Drop unused columns or keep what you want for features
feature_cols = ['GridPosition', 'StatusEncoded', 'FinishedFlag', 'LapsRatio', 'DriverEncoded', 'TeamEncoded', 'Year']

X = df[feature_cols]

# Example: if predicting finishing position, y = df['Position']

print(X.head())

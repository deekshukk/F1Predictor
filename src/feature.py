import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features_and_target(csv_path='data/processed/f1_2025_multiple_races.csv'):
    # === 1. Load the CSV ===
    df = pd.read_csv(csv_path)

    # === 2. Sort to preserve time order ===
    df = df.sort_values(by=['Year', 'Race', 'Driver']).reset_index(drop=True)

    # === 3. Encode categorical columns ===
    le_team = LabelEncoder()
    le_driver = LabelEncoder()
    le_race = LabelEncoder()

    df['Team_enc'] = le_team.fit_transform(df['Team'])
    df['Driver_enc'] = le_driver.fit_transform(df['Driver'])
    df['Race_enc'] = le_race.fit_transform(df['Race'])

    # === 4. Add Race number in season ===
    race_order = df[['Year', 'Race', 'Race_enc']].drop_duplicates().sort_values(['Year', 'Race_enc'])
    race_order['Race_number'] = range(1, len(race_order) + 1)
    df = df.merge(race_order[['Year', 'Race', 'Race_number']], on=['Year', 'Race'], how='left')

    # === 5. Convert columns to numeric ===
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)

    # === 6. Rolling driver stats ===
    df['Driver_avg_finish'] = (
        df.groupby('Driver')['Position']
          .transform(lambda x: x.shift().expanding().mean())
    )

    df['Driver_cum_points'] = (
        df.groupby('Driver')['Points']
          .transform(lambda x: x.shift().cumsum())
    )

    df['Driver_finish_std'] = (
        df.groupby('Driver')['Position']
          .transform(lambda x: x.shift().expanding().std())
    )

    # === 7. Team rolling average ===
    df['Team_avg_finish'] = (
        df.groupby('Team')['Position']
          .transform(lambda x: x.shift().expanding().mean())
    )

    # === 8. Grid vs Finish (gain/loss) ===
    df['Grid_vs_Finish'] = df['GridPosition'] - df['Position']

    # === 9. Binary status (Finished or not) ===
    df['Finished'] = df['Status'].apply(lambda x: 1 if str(x).lower() == 'finished' else 0)

    # === 10. Handle missing values ===
    df['Driver_avg_finish'] = df['Driver_avg_finish'].fillna(df['Position'].mean())
    df['Driver_cum_points'] = df['Driver_cum_points'].fillna(0)
    df['Driver_finish_std'] = df['Driver_finish_std'].fillna(0)
    df['Team_avg_finish'] = df['Team_avg_finish'].fillna(df['Position'].mean())
    df['Grid_vs_Finish'] = df['Grid_vs_Finish'].fillna(0)

    # === 11. Select features and target ===
    features = df[[
        'GridPosition',
        'Team_enc',
        'Driver_enc',
        'Race_enc',
        'Race_number',
        'Driver_avg_finish',
        'Driver_cum_points',
        'Driver_finish_std',
        'Team_avg_finish',
        'Grid_vs_Finish',
        'Finished'
    ]]

    target = df['Position']

    return features, target

# If running directly, print samples
if __name__ == "__main__":
    X, y = create_features_and_target()
    print("Feature sample:")
    print(X.head())
    print("\nTarget sample:")
    print(y.head())

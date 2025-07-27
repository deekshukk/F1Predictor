import fastf1
import pandas as pd
import os

def collect_multiple_races(year, races):
    fastf1.Cache.enable_cache('cache')  # Cache folder for fastf1 data

    all_races = []

    for race_name in races:
        print(f"Loading {race_name} {year} race data...")
        session = fastf1.get_session(year, race_name, 'R')  # 'R' for race
        session.load()
        results = session.results

        # Select relevant columns and rename
        df = results[['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points', 'Status', 'Laps']].copy()
        df.rename(columns={
            'Abbreviation': 'Driver',
            'TeamName': 'Team'
        }, inplace=True)
        df['Race'] = race_name
        df['Year'] = year

        all_races.append(df)

    # Combine all races into one DataFrame
    combined_df = pd.concat(all_races, ignore_index=True)

    # Ensure folder exists
    os.makedirs('data/processed', exist_ok=True)
    save_path = f'data/processed/f1_{year}_multiple_races.csv'
    combined_df.to_csv(save_path, index=False)
    print(f"Saved combined race data to {save_path}")

if __name__ == "__main__":
    races_2025 = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan',
    'Miami', 'Monaco', 'Spain', 'Canada', 'Austria',
    'Britain', 'Hungary', 'Belgium'
    ]  
    collect_multiple_races(2025, races_2025)

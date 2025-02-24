import sys
import understatapi
import pandas as pd
from datetime import datetime
from build_sportsmonks import build_sportsmonks_data

class DatasetBuilder:
    def __init__(self, league="EPL", season="2024"):
        self.league = league
        self.season = season

    def build_understat_data(self):
        """
        Builds a dataset from Understat API data.
        Returns a DataFrame with match data from both home and away team perspectives.
        """
        understat = understatapi.UnderstatClient()
        try:
            matches = understat.league(league=self.league).get_match_data(season=self.season)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

        rows = []
        for match in matches:
            if not match['isResult']:  # Skip matches that haven't been played
                continue
                
            # Parse datetime
            dt = datetime.strptime(match["datetime"], "%Y-%m-%d %H:%M:%S")
            
            # Create base match data that's same for both teams
            base_data = {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "day": dt.strftime("%A"),
                "season": self.season
            }
            
            # Determine match result
            home_goals = int(match["goals"]["h"])
            away_goals = int(match["goals"]["a"])
            
            if home_goals > away_goals:
                home_result = 'W'
                away_result = 'L'
            elif home_goals < away_goals:
                home_result = 'L'
                away_result = 'W'
            else:
                home_result = away_result = 'D'
            
            # Home team row
            home_row = base_data.copy()
            home_row.update({
                "team": match["h"]["title"],
                "opponent": match["a"]["title"],
                "gf": home_goals,
                "ga": away_goals,
                "xg": float(match["xG"]["h"]),
                "xga": float(match["xG"]["a"]),
                "venue": "home",
                "result": home_result
            })
            rows.append(home_row)
            
            # Away team row
            away_row = base_data.copy()
            away_row.update({
                "team": match["a"]["title"],
                "opponent": match["h"]["title"],
                "gf": away_goals,
                "ga": home_goals,
                "xg": float(match["xG"]["a"]),
                "xga": float(match["xG"]["h"]),
                "venue": "away",
                "result": away_result
            })
            rows.append(away_row)
        
        if not rows:
            print("No match data found.")
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=[
            "date", "time", "day", "team", "opponent",
            "gf", "ga", "xg", "xga",
            "venue", "result", "season"
        ])

    def build_offline(self):
        """
        Builds the latest dataset.
        Final prediction model will have three parts:
        1. A classic ML model
        2. A deep learning model
        3. An agent that can access a vector database of soccer stats

        this build function will be the dataset for the classic ML model and the deep learning model
        """
        print('Building the latest dataset...')
        df = self.build_understat_data()
        # df = build_sportsmonks_data()
        if not df.empty:
            df.to_csv(f'{self.league}_data_{self.season}.csv', index=False)
            print('Dataset built successfully.')
        else:
            print('No data to save.')

    def build_online(self):
        """
        Builds the latest dataset.
        """
        print('Building the latest online dataset...')
        # Implement logic to fetch and update the online dataset

if __name__ == '__main__':
    builder = DatasetBuilder()
    if len(sys.argv) < 2:
        print("Usage: python latest_dataset_builder.py [offline|online]") 
    elif sys.argv[1] == "offline":
        builder.build_offline()
    elif sys.argv[1] == "online":
        builder.build_online()
    else:
        print("Invalid argument. Use 'offline' or 'online'.")
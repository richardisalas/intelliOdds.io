import sys
import understatapi
import asyncio
import json
import aiohttp
import pandas as pd
from datetime import datetime
from build_sportsmonks import build_sportsmonks_data
from understat import Understat

class DatasetBuilder:
    def __init__(self, league="EPL", season="2024"):
        self.league = league
        self.season = season
    
    async def get_match_stats(self, match_id):
        """
        Fetch detailed match statistics for a given match ID
        """
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            try:
                match_stats = await understat.get_match_stats(match_id)
                return match_stats
            except Exception as e:
                print(f"Error fetching match stats for match {match_id}: {e}")
                return None

    async def build_understat_data(self):
        """
        Builds a dataset from Understat API data.
        Returns a DataFrame with match data from both home and away team perspectives.
        """
        under_stat = understatapi.UnderstatClient()
        try:
            matches = under_stat.league(league=self.league).get_match_data(season=self.season)
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
                "time": dt.strftime("%H:%M"),
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
            
            # Get detailed match statistics
            match_stats = await self.get_match_stats(match["id"])
            
            # Extract statistics if available
            stats = {}
            if match_stats:
                try:
                    stats = {
                        'home_shots': int(match_stats.get('h_shot', 0)),
                        'away_shots': int(match_stats.get('a_shot', 0)),
                        'home_shots_on_target': int(match_stats.get('h_shotOnTarget', 0)),
                        'away_shots_on_target': int(match_stats.get('a_shotOnTarget', 0)),
                        'home_deep': int(match_stats.get('h_deep', 0)),
                        'away_deep': int(match_stats.get('a_deep', 0)),
                        'home_ppda': float(match_stats.get('h_ppda', 0)),
                        'away_ppda': float(match_stats.get('a_ppda', 0))
                    }
                except Exception as e:
                    print(f"Error processing match stats for match {match['id']}: {e}")
            
            # Home team row
            home_row = base_data.copy()
            home_row.update({
                "team": match["h"]["title"], # short title for better embedding?
                "opponent": match["a"]["title"],
                "gf": home_goals,
                "ga": away_goals,
                "xg": float(match["xG"]["h"]),
                "xga": float(match["xG"]["a"]),
                "venue": "home",
                "result": home_result,
                "shots": stats.get('home_shots', 0),
                "shots_on_target": stats.get('home_shots_on_target', 0),
                "deep": stats.get('home_deep', 0),
                "ppda": stats.get('home_ppda', 0)
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
                "result": away_result,
                "shots": stats.get('away_shots', 0),
                "shots_on_target": stats.get('away_shots_on_target', 0),
                "deep": stats.get('away_deep', 0),
                "ppda": stats.get('away_ppda', 0)
            })
            rows.append(away_row)
        
        if not rows:
            print("No match data found.")
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=[
            "date", "time", "day", "team", "opponent",
            "gf", "ga", "xg", "xga",
            "venue", "result", "shots", "shots_on_target",
            "deep", "ppda", "season"
        ])

    async def build_offline_async(self):
        """
        Builds the latest dataset.
        Final prediction model will have three parts:
        1. A classic ML model
        2. A deep learning model
        3. An agent that can access a vector database of soccer stats

        this build function will be the dataset for the classic ML model and the deep learning model
        """
        print('Building the latest dataset...')
        df = await self.build_understat_data()
        if not df.empty:
            df.to_csv(f'{self.league}_data_{self.season}.csv', index=False)
            print('Dataset built successfully.')
        else:
            print('No data to save.')

    def build_offline(self):
        """
        Synchronous wrapper for build_offline_async
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.build_offline_async())

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
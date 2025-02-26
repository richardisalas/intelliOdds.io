import sys
import understatapi
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from understat import Understat
import unicodedata
import html

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

    def analyze_close_shots(self, shots_data, team_type):
        """
        Analyze shots within 20 meters of goal.
        Args:
            shots_data (dict): Dictionary containing shot data
            team_type (str): 'h' for home or 'a' for away
        Returns:
            tuple: (number of close shots, number of close shots on target)
        """
        close_shots = 0
        close_shots_on_target = 0
        
        # Convert 20 meters to X coordinate (pitch coordinates are normalized)
        CLOSE_SHOT_THRESHOLD = 0.8  # 20 meters from goal is approximately 0.8 in normalized coordinates
        
        for shot in shots_data[team_type]:
            shot_x = float(shot['X'])
            # Check if shot is within 20 meters
            if shot_x >= CLOSE_SHOT_THRESHOLD:
                close_shots += 1
                # Check if shot was on target
                if shot['result'] in ['Goal', 'SavedShot']:
                    close_shots_on_target += 1
        
        return close_shots, close_shots_on_target

    def analyze_roster_data(self, roster_data, team_side):
        """
        Analyzes roster data for a specific team (home 'h' or away 'a')
        Returns a dictionary of aggregated team statistics
        """
        team_data = roster_data.get(team_side, {})
        
        # Initialize aggregates
        total_xGChain = 0
        total_xGBuildup = 0
        total_key_passes = 0
        total_xA = 0
        yellow_cards = 0
        red_cards = 0
        substitutions = 0
        
        # Track starting XI stats
        starting_xi_count = 0
        avg_position_order = 0
        
        for player in team_data.values():
            # Sum up team totals
            total_xGChain += float(player.get('xGChain', 0))
            total_xGBuildup += float(player.get('xGBuildup', 0))
            total_key_passes += int(player.get('key_passes', 0))
            total_xA += float(player.get('xA', 0))
            yellow_cards += int(player.get('yellow_card', 0))
            red_cards += int(player.get('red_card', 0))
            
            # Count substitutions
            if player.get('roster_in') != '0' or player.get('roster_out') != '0':
                substitutions += 1
            
            # Track starting XI
            if player.get('roster_in') == '0' and player.get('position') != 'Sub':
                starting_xi_count += 1
                avg_position_order += float(player.get('positionOrder', 0))
        
        # Calculate average position order for starting XI
        avg_position_order = avg_position_order / starting_xi_count if starting_xi_count > 0 else 0
        
        return {
            'xGChain': total_xGChain,
            'xGBuildup': total_xGBuildup,
            'key_passes': total_key_passes,
            'xA': total_xA,
            'yellow_cards': yellow_cards,
            'red_cards': red_cards,
            'substitutions': substitutions // 2,  # Divide by 2 since each sub counts as in/out
            'avg_position_order': avg_position_order
        }

    def decode_and_remove_accents(self, input_str):
        # 1) Decode HTML entities like &#039; -> '
        decoded_str = html.unescape(input_str)

        # 2) Normalize and strip accents if there are any
        normalized_str = unicodedata.normalize('NFKD', decoded_str)
        ascii_str = normalized_str.encode('ASCII', 'ignore').decode('ASCII')

        return ascii_str

    def get_starting_xi(self, roster_data, team_side):
        """
        Extracts the starting XI players in order of their position (GK first, then outfield players).
        Args:
            roster_data (dict): The roster data dictionary
            team_side (str): 'h' for home or 'a' for away
        Returns:
            dict: Dictionary with GK and player_1 through player_10 keys
        """
        team_data = roster_data.get(team_side, {})
        starting_xi = []

        # Get all players (up to 11)
        for player in team_data.values():
            starting_xi.append({
                'name': self.decode_and_remove_accents(player.get('player', '')),
                'position_order': float(player.get('positionOrder', 99))
            })
            if len(starting_xi) == 11:  # Limit to 11 players
                break

        # Sort players by position order
        starting_xi.sort(key=lambda x: x['position_order'])

        # Create the result dictionary
        result = {}
        
        # Assign the first player as the goalkeeper
        result['GK'] = starting_xi[0]['name'] if starting_xi else 'Unknown'
        
        # Assign the next 10 players as outfield players
        for i in range(1, 11):
            result[f'player_{i}'] = starting_xi[i]['name'] if i < len(starting_xi) else 'Unknown'
        
        return result

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
            
            # Get roster data and analyze it
            roster_data = under_stat.match(match["id"]).get_roster_data()
            home_roster_stats = self.analyze_roster_data(roster_data, 'h')
            away_roster_stats = self.analyze_roster_data(roster_data, 'a')
            
            # Get starting XI for both teams
            home_starting_xi = self.get_starting_xi(roster_data, 'h')
            away_starting_xi = self.get_starting_xi(roster_data, 'a')

            # Get shot data
            shot_data = under_stat.match(match["id"]).get_shot_data()
            
            # Analyze close shots for both teams
            home_close_shots, home_close_shots_on_target = self.analyze_close_shots(shot_data, 'h')
            away_close_shots, away_close_shots_on_target = self.analyze_close_shots(shot_data, 'a')

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
                "team": match["h"]["title"],
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
                "ppda": stats.get('home_ppda', 0),
                "close_shots": home_close_shots,
                "close_shots_on_target": home_close_shots_on_target,
                "xGChain": home_roster_stats['xGChain'],
                "xGBuildup": home_roster_stats['xGBuildup'],
                "key_passes": home_roster_stats['key_passes'],
                "xA": home_roster_stats['xA'],
                "yellow_cards": home_roster_stats['yellow_cards'],
                "red_cards": home_roster_stats['red_cards'],
                "substitutions": home_roster_stats['substitutions'],
                "avg_position_order": home_roster_stats['avg_position_order']
            })
            # Add starting XI to home row
            home_row.update(home_starting_xi)
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
                "ppda": stats.get('away_ppda', 0),
                "close_shots": away_close_shots,
                "close_shots_on_target": away_close_shots_on_target,
                "xGChain": away_roster_stats['xGChain'],
                "xGBuildup": away_roster_stats['xGBuildup'],
                "key_passes": away_roster_stats['key_passes'],
                "xA": away_roster_stats['xA'],
                "yellow_cards": away_roster_stats['yellow_cards'],
                "red_cards": away_roster_stats['red_cards'],
                "substitutions": away_roster_stats['substitutions'],
                "avg_position_order": away_roster_stats['avg_position_order']
            })
            # Add starting XI to away row
            away_row.update(away_starting_xi)
            rows.append(away_row)
        
        if not rows:
            print("No match data found.")
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=[
            "date", "time", "day", "team", "opponent",
            "gf", "ga", "xg", "xga",
            "venue", "result", "shots", "shots_on_target",
            "deep", "ppda", "close_shots", "close_shots_on_target",
            "xGChain", "xGBuildup", "key_passes", "xA",
            "yellow_cards", "red_cards", "substitutions", "avg_position_order",
            "GK", "player_1", "player_2", "player_3", "player_4", "player_5",
            "player_6", "player_7", "player_8", "player_9", "player_10",
            "season"
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
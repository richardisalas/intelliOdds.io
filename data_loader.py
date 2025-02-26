import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Union, Optional
from pathlib import Path
from utils import DataUtils

class DataLoader:
    """
    A class to load and preprocess soccer match data for machine learning models.
    Provides flexible feature selection and preprocessing options.
    """
    
    def __init__(self, data_path: str, league: str = "EPL", season: str = "2024"):
        """
        Initialize the DataLoader with path to data and optional filters.
        
        Args:
            data_path (str): Path to the directory containing the dataset
            league (str): League name (default: "EPL")
            season (str): Season year (default: "2024")
        """
        self.data_path = Path(data_path)
        self.league = league
        self.season = season
        self.file_name = f"{league}_data_{season}.csv"
        self.data = None
        self.feature_encoders = {}
        self.scaler = None
        
        # Define feature groups for easy selection
        self.feature_groups = {
            'basic': [
                'team', 'opponent', 'venue', 'result'
            ],
            'match_stats': [
                'gf', 'ga', 'xg', 'xga',
                'shots', 'shots_on_target', 'deep', 'ppda',
                'close_shots', 'close_shots_on_target'
            ],
            'advanced_stats': [
                'xGChain', 'xGBuildup', 'key_passes', 'xA',
                'yellow_cards', 'red_cards', 'substitutions',
                'avg_position_order'
            ],
            'lineup': [
                'GK', 'player_1', 'player_2', 'player_3', 'player_4',
                'player_5', 'player_6', 'player_7', 'player_8',
                'player_9', 'player_10'
            ],
            'temporal': ['date', 'time', 'day', 'season']
        }
        
        # Define which features should be encoded vs scaled
        self.categorical_features = [
            'team', 'opponent', 'venue', 'result', 'day',
            'GK', 'player_1', 'player_2', 'player_3', 'player_4',
            'player_5', 'player_6', 'player_7', 'player_8',
            'player_9', 'player_10'
        ]
        
        self.numerical_features = [
            'gf', 'ga', 'xg', 'xga', 'shots', 'shots_on_target',
            'deep', 'ppda', 'close_shots', 'close_shots_on_target',
            'xGChain', 'xGBuildup', 'key_passes', 'xA',
            'yellow_cards', 'red_cards', 'substitutions',
            'avg_position_order'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from file.
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        file_path = self.data_path / self.file_name
        try:
            self.data = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            # Fill missing values immediately after loading
            self.fill_missing_values()
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"No data file found at {file_path}")
    
    def fill_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """
        Fill missing values in the dataset using the DataUtils class.
        
        Args:
            strategy (str): Strategy to use for filling values (default: 'median')
        
        Returns:
            pd.DataFrame: The dataset with missing values filled
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use the DataUtils class for filling missing values
        self.data = DataUtils.fill_missing_values(
            df=self.data,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            strategy=strategy
        )
        
        return self.data
    
    def get_feature_list(self, groups: Optional[List[str]] = None) -> List[str]:
        """
        Get list of features based on specified groups.
        
        Args:
            groups (List[str], optional): List of feature groups to include
            
        Returns:
            List[str]: List of features from specified groups
        """
        if groups is None:
            groups = list(self.feature_groups.keys())
            
        features = []
        for group in groups:
            if group in self.feature_groups:
                features.extend(self.feature_groups[group])
            else:
                print(f"Warning: Unknown feature group '{group}'")
        
        return list(set(features))  # Remove duplicates
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """
        Get all available features grouped by category.
        
        Returns:
            Dict[str, List[str]]: Dictionary of feature groups and their features
        """
        return self.feature_groups
        
    def _get_preprocessed_file_path(self) -> Path:
        """
        Get the path for the preprocessed data file.
        
        Returns:
            Path: Path object for the preprocessed file
        """
        base_name = self.file_name.rsplit('.', 1)[0]
        return self.data_path / f"{base_name}_preprocessed.csv"
        
    def save_preprocessed_data(self, df: pd.DataFrame) -> None:
        """
        Save the preprocessed dataset to a CSV file.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame to save
        """
        preprocessed_path = self._get_preprocessed_file_path()
        df.to_csv(preprocessed_path, index=False)
        print(f"Saved preprocessed data to {preprocessed_path}")
        
    def load_or_create_preprocessed_data(self, 
                                       small_window: int = 3, 
                                       window: int = 5,
                                       season_matches: int = 38) -> pd.DataFrame:
        """
        Load preprocessed data if it exists, otherwise create and save it.
        This method follows a sequential approach that avoids duplicating calculations.
        
        Args:
            small_window (int): Size of small rolling window (default: 3)
            window (int): Size of standard rolling window (default: 5)
            season_matches (int): Total matches in a season (default: 38)
            
        Returns:
            pd.DataFrame: DataFrame with all preprocessed features
        """
        preprocessed_path = self._get_preprocessed_file_path()
        
        # Try to load existing preprocessed data
        if preprocessed_path.exists():
            print(f"Loading existing preprocessed data from {preprocessed_path}")
            try:
                preprocessed_df = pd.read_csv(preprocessed_path)
                # Convert date column back to datetime
                if 'date' in preprocessed_df.columns:
                    preprocessed_df['date'] = pd.to_datetime(preprocessed_df['date'])
                return preprocessed_df
            except Exception as e:
                print(f"Error loading preprocessed data: {e}")
                print("Regenerating preprocessed data...")
        else:
            print("No existing preprocessed data found. Creating new preprocessed dataset...")
            
        # Load raw data if not already loaded
        if self.data is None:
            self.load_data()
            
        # Step 1: First create time series features
        print("Creating time series features...")
        df_with_features = self.create_time_series_features(
            small_window=small_window,
            window=window
        )
        
        # Step 2: Pass that DataFrame to the advanced features method
        print("Creating advanced features...")
        final_df = self.create_advanced_features(
            df=df_with_features,
            small_window=small_window,
            window=window,
            season_matches=season_matches
        )
        
        # Save the preprocessed data
        self.save_preprocessed_data(final_df)
        
        return final_df

    def preprocess_data(self, 
                       feature_groups: Optional[List[str]] = ['basic', 'match_stats', 'advanced_stats', 'temporal', 'lineup'],
                       additional_features: Optional[List[str]] = None,
                       exclude_features: Optional[List[str]] = None,
                       target_variable: str = 'result',
                       scale_features: bool = True,
                       force_reprocess: bool = False) -> tuple:
        """
        Preprocess the data for machine learning.
        
        Args:
            feature_groups (List[str], optional): Groups of features to include
                                                    (default: all available feature groups)
            additional_features (List[str], optional): Additional individual features to include
            exclude_features (List[str], optional): Features to exclude
            target_variable (str): Target variable for prediction (default: 'result')
            scale_features (bool): Whether to scale numerical features (default: True)
            force_reprocess (bool): Whether to force reprocessing even if preprocessed data exists
            
        Returns:
            tuple: (X, y) preprocessed features and target
        """
        # Load or create preprocessed data
        if not force_reprocess:
            self.data = self.load_or_create_preprocessed_data()
        else:
            print("Forcing reprocessing of data...")
            preprocessed_path = self._get_preprocessed_file_path()
            if preprocessed_path.exists():
                preprocessed_path.unlink()  # Delete existing preprocessed file
            self.data = self.load_or_create_preprocessed_data()
            
        # Continue with feature selection and scaling
        features = self.get_feature_list(feature_groups)
        
        # Add additional features if specified
        if additional_features:
            features.extend(additional_features)
            
        # Remove excluded features
        if exclude_features:
            features = [f for f in features if f not in exclude_features]
            
        # Remove target variable from features if it's there
        if target_variable in features:
            features.remove(target_variable)
            
        # Create feature matrix and target
        X = self.data[features].copy()
        y = self.data[target_variable].copy()
        
        # Encode categorical variables using DataUtils
        categorical_cols = [f for f in features if f in self.categorical_features]
        if categorical_cols:
            X, self.feature_encoders = DataUtils.encode_categorical_features(
                df=X,
                categorical_features=categorical_cols,
                encoders=self.feature_encoders
            )
        
        # Scale numerical features if requested
        if scale_features:
            numerical_cols = [col for col in features if col in self.numerical_features]
            if numerical_cols:
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
                else:
                    X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X, y
    
    def create_time_series_features(self, small_window: int = 3, window: int = 5) -> pd.DataFrame:
        """
        Create time series features by adding rolling averages and trends.
        Uses data from PAST matches only to avoid data leakage.
        Computes two sets of rolling averages with different window sizes.
        
        Args:
            small_window (int): Size of the smaller rolling window for recent form (default: 3)
            window (int): Size of the standard rolling window (default: 5)
            
        Returns:
            pd.DataFrame: DataFrame with additional time series features
        """
        if self.data is None:
            self.load_data()
            
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by team and create rolling features from PAST matches only
        new_features = []
        for team in df['team'].unique():
            # Get data for this team and sort by date
            team_data = df[df['team'] == team].sort_values('date')
            
            # Create a separate DataFrame for calculations to avoid leakage
            team_calc = team_data.copy()
            
            # Convert result to points: W=3, D=1, L=0
            team_calc['match_points'] = team_calc['result'].map({'W': 3, 'D': 1, 'L': 0})

            # Calculate cumulative sum of points
            team_calc['cumulative_points'] = team_calc['match_points'].cumsum()

            # Shift to ensure we only use past matches (avoid data leakage)
            team_calc['cumulative_points'] = team_calc['cumulative_points'].shift(1)

            # First match should have 0 points
            team_calc['cumulative_points'] = team_calc['cumulative_points'].fillna(0)
            
            # Calculate rolling averages with both window sizes
            for feature in self.numerical_features:
                # Small window rolling average
                small_col_name = f'{feature}_rolling_avg_{small_window}'
                team_calc[small_col_name] = team_calc[feature].rolling(
                    window=small_window, min_periods=1).mean().shift(1)
                team_calc[small_col_name] = team_calc[small_col_name].fillna(0)
                
                # Standard window rolling average
                std_col_name = f'{feature}_rolling_avg_{window}'
                team_calc[std_col_name] = team_calc[feature].rolling(
                    window=window, min_periods=1).mean().shift(1)
                team_calc[std_col_name] = team_calc[std_col_name].fillna(0)
                
            # Calculate recent points in last 3 and 5 matches
            team_calc[f'points_last_{small_window}'] = team_calc['match_points'].rolling(
                window=small_window, min_periods=1).sum().shift(1)
            team_calc[f'points_last_{small_window}'] = team_calc[f'points_last_{small_window}'].fillna(0)
            
            team_calc[f'points_last_{window}'] = team_calc['match_points'].rolling(
                window=window, min_periods=1).sum().shift(1)
            team_calc[f'points_last_{window}'] = team_calc[f'points_last_{window}'].fillna(0)
            
            # Create recent opponent strength features for both windows
            # Small window opponent strength
            team_calc[f'opponent_strength_{small_window}'] = team_calc['opponent'].map(
                df.groupby('team')['xg'].mean()).rolling(window=small_window, min_periods=1).mean().shift(1)
            team_calc[f'opponent_strength_{small_window}'] = team_calc[f'opponent_strength_{small_window}'].fillna(df['xg'].mean())
            
            # Standard window opponent strength
            team_calc[f'opponent_strength_{window}'] = team_calc['opponent'].map(
                df.groupby('team')['xg'].mean()).rolling(window=window, min_periods=1).mean().shift(1)
            team_calc[f'opponent_strength_{window}'] = team_calc[f'opponent_strength_{window}'].fillna(df['xg'].mean())
            
            # Save the calculated features
            new_features.append(team_calc)
        
        # Combine all teams back together
        result_df = pd.concat(new_features)
        
        # Sort back to original order
        result_df = result_df.sort_index()
        
        # Clean up intermediate calculation columns
        if 'match_points' in result_df.columns:
            result_df = result_df.drop(columns=['match_points'])
        
        return result_df
    
    def get_feature_importance(self, model, feature_groups: Optional[List[str]] = None) -> Dict:
        """
        Get feature importance scores for a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_groups (List[str], optional): Groups of features to include
            
        Returns:
            Dict: Dictionary of feature importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")
            
        features = self.get_feature_list(feature_groups)
        importance_dict = dict(zip(features, model.feature_importances_))
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def create_advanced_features(self, df: Optional[pd.DataFrame] = None, small_window: int = 3, 
                               window: int = 5, season_matches: int = 38) -> pd.DataFrame:
        """
        Create advanced analytical features for football match prediction.
        Can either start with raw data or build upon an existing DataFrame with time series features.
        All calculations use ONLY past data to avoid data leakage.
        
        Args:
            df (pd.DataFrame, optional): Existing DataFrame with time series features
            small_window (int): Size of small rolling window (default: 3)
            window (int): Size of standard rolling window (default: 5)
            season_matches (int): Total matches in a season (default: 38)
            
        Returns:
            pd.DataFrame: DataFrame with advanced analytical features
        """
        # If no DataFrame is provided, start with raw data
        if df is None:
            if self.data is None:
                self.load_data()
            df = self.data.copy()
            # Calculate cumulative points as needed for advanced features
            df = self._calculate_basic_time_features(df)
        else:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Pre-calculate goal differential for all matches
        if 'goal_diff' not in df.columns:
            df['goal_diff'] = df['gf'] - df['ga']
        
        # Group by team and create advanced features from PAST matches only
        advanced_features = []
        for team in df['team'].unique():
            # Get data for this team and sort by date
            team_data = df[df['team'] == team].sort_values('date')
            
            # Create a separate DataFrame for calculations to avoid leakage
            team_calc = team_data.copy()
            
            # We don't need to recalculate cumulative_points if it exists
            if 'cumulative_points' not in team_calc.columns:
                # Calculate cumulative points
                team_calc['match_points'] = team_calc['result'].map({'W': 3, 'D': 1, 'L': 0})
                team_calc['cumulative_points'] = team_calc['match_points'].cumsum()
                team_calc['cumulative_points'] = team_calc['cumulative_points'].shift(1)
                team_calc['cumulative_points'] = team_calc['cumulative_points'].fillna(0)
            
            # 1. Goal Difference Momentum
            # Calculate rolling goal differentials
            team_calc['gd_rolling_small'] = team_calc['goal_diff'].rolling(window=small_window, min_periods=2).mean()
            team_calc['gd_rolling_std'] = team_calc['goal_diff'].rolling(window=window, min_periods=2).mean()
            
            # Calculate the momentum (slope of the goal differential trend)
            team_calc['gd_momentum'] = team_calc.apply(
                lambda row: self._calculate_gd_slope(
                    team_calc.loc[:row.name], small_window
                ) if team_calc.loc[:row.name].shape[0] >= small_window else 0,
                axis=1
            )
            # Shift to ensure we only use past data
            team_calc['gd_momentum'] = team_calc['gd_momentum'].shift(1).fillna(0)
            
            # 2. Home/Away Performance Disparity
            # Create home/away indicators
            team_calc['is_home'] = (team_calc['venue'] == 'home').astype(int)
            team_calc['is_away'] = (team_calc['venue'] == 'away').astype(int)
            
            # Use existing match_points if available, otherwise calculate
            if 'match_points' not in team_calc.columns:
                team_calc['match_points'] = team_calc['result'].map({'W': 3, 'D': 1, 'L': 0})
            
            # Calculate home and away points separately
            team_calc['home_points'] = team_calc['match_points'] * team_calc['is_home']
            team_calc['away_points'] = team_calc['match_points'] * team_calc['is_away']
            
            # Calculate rolling sums for home and away (past 10 matches)
            home_away_window = 10
            team_calc['last10_home_points'] = team_calc['home_points'].rolling(window=home_away_window, min_periods=1).sum().shift(1)
            team_calc['last10_away_points'] = team_calc['away_points'].rolling(window=home_away_window, min_periods=1).sum().shift(1)
            team_calc['last10_home_matches'] = team_calc['is_home'].rolling(window=home_away_window, min_periods=1).sum().shift(1)
            team_calc['last10_away_matches'] = team_calc['is_away'].rolling(window=home_away_window, min_periods=1).sum().shift(1)
            
            # Calculate average points per match for home and away
            team_calc['home_ppg'] = team_calc.apply(
                lambda row: row['last10_home_points'] / row['last10_home_matches'] 
                if row['last10_home_matches'] > 0 else 0,
                axis=1
            )
            team_calc['away_ppg'] = team_calc.apply(
                lambda row: row['last10_away_points'] / row['last10_away_matches']
                if row['last10_away_matches'] > 0 else 0,
                axis=1
            )
            
            # Calculate home/away disparity (ratio)
            team_calc['home_away_disparity'] = team_calc.apply(
                lambda row: row['home_ppg'] / row['away_ppg'] 
                if row['away_ppg'] > 0 else (2.0 if row['home_ppg'] > 0 else 1.0),
                axis=1
            )
            
            # 3. Match Importance Coefficient
            # Calculate match number within the season
            team_calc['match_number'] = team_calc.groupby(pd.Grouper(key='date', freq='YE')).cumcount() + 1
            
            # Calculate remaining matches in the season
            team_calc['remaining_matches'] = season_matches - team_calc['match_number']
            
            # Calculate league position (using cumulative points as a proxy)
            # In a real implementation, you would need to calculate actual table position
            team_calc['league_position_proxy'] = team_calc['cumulative_points'].rank(ascending=False, method='dense')
            
            # Calculate match importance coefficient
            # Higher when fewer matches remain and team is close to important positions (1, 4, 17)
            team_calc['match_importance'] = team_calc.apply(
                lambda row: self._calculate_match_importance(
                    row['league_position_proxy'],
                    row['remaining_matches'],
                    season_matches
                ),
                axis=1
            )
            # Shift to ensure we only use past information
            team_calc['match_importance'] = team_calc['match_importance'].shift(1).fillna(0.5)
            
            # 4. Rest Day Advantage
            # Calculate days since last match
            team_calc['days_since_last_match'] = team_calc['date'].diff().dt.days
            # Fill NA for first match
            team_calc['days_since_last_match'] = team_calc['days_since_last_match'].fillna(7)  # Assume typical week rest for first match
            
            # We need to compare against opponent's rest days
            # First, save this team's rest days in a lookup dict for later use
            team_rest_days = dict(zip(team_calc['date'], team_calc['days_since_last_match']))
            
            # Store this information for later when we combine teams
            team_calc['_team_rest_days'] = team_calc['days_since_last_match']
            team_calc['_team'] = team
            
            # Clean up intermediate columns
            cols_to_keep = [col for col in team_calc.columns if not col.startswith('last10_') and 
                           col not in ['home_points', 'away_points', 'is_home', 'is_away', 'match_number']]
            team_calc = team_calc[cols_to_keep]
            
            # Save the calculated features
            advanced_features.append(team_calc)
        
        # Combine all teams back together
        result_df = pd.concat(advanced_features)
        
        # Calculate rest day advantage (requires data from all teams)
        # For each match, find the opponent's rest days and calculate the difference
        result_df['opponent_rest_days'] = result_df.apply(
            lambda row: self._get_opponent_rest_days(
                result_df, row['date'], row['opponent'], row['_team']
            ),
            axis=1
        )
        result_df['rest_day_advantage'] = result_df['_team_rest_days'] - result_df['opponent_rest_days']
        
        # Clean up intermediate columns
        cols_to_drop = ['_team_rest_days', '_team', 'opponent_rest_days', 'remaining_matches']
        result_df = result_df.drop(columns=cols_to_drop)
        
        # Sort back to original order
        result_df = result_df.sort_index()
        
        return result_df

    def _calculate_gd_slope(self, team_history, window):
        """
        Calculate the slope of recent goal differentials to measure momentum.
        A positive slope indicates improving performance, negative indicates decline.
        
        Args:
            team_history (pd.DataFrame): Team's match history
            window (int): Number of recent matches to consider
            
        Returns:
            float: Slope of the goal differential trend
        """
        recent_matches = team_history.tail(window)
        if len(recent_matches) < 2:  # Need at least 2 points for a slope
            return 0
            
        x = np.arange(len(recent_matches))
        y = recent_matches['goal_diff'].values
        
        # Handle cases with insufficient data
        if len(x) != len(y) or len(x) < 2:
            return 0
            
        # Linear regression to find slope
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            return 0
    
    def _calculate_match_importance(self, league_position, remaining_matches, season_matches):
        """
        Calculate how important a match is based on league position and matches remaining.
        
        Args:
            league_position (float): Team's position in the league
            remaining_matches (int): Number of matches remaining in the season
            season_matches (int): Total matches in a season
            
        Returns:
            float: Match importance coefficient between 0 and 1
        """
        # Season progress factor (higher toward end of season)
        season_progress = 1 - (remaining_matches / season_matches)
        season_factor = 0.5 + (0.5 * season_progress**2)  # Quadratic increase in importance
        
        # Position factor (higher for positions near title, Champions League spots, relegation)
        # Important positions in EPL: 1 (title), 4 (Champions League), 17 (relegation)
        position_importance = 0
        
        if league_position <= 2:  # Title contention
            position_importance = 1.0
        elif league_position <= 5:  # Champions League spots
            position_importance = 0.8
        elif league_position <= 7:  # Europa League spots
            position_importance = 0.6
        elif league_position >= 15:  # Relegation battle
            position_importance = 0.7 + (0.3 * (league_position - 15) / 5)
        else:  # Mid-table
            position_importance = 0.3
            
        return position_importance * season_factor
    
    def _get_opponent_rest_days(self, all_teams_df, match_date, opponent, team):
        """
        Find the opponent's rest days for a specific match.
        
        Args:
            all_teams_df (pd.DataFrame): DataFrame with all teams' data
            match_date (datetime): Date of the match
            opponent (str): Name of the opponent
            team (str): Current team
            
        Returns:
            float: Number of rest days for the opponent
        """
        # Find the opponent's match on the same date
        opponent_match = all_teams_df[
            (all_teams_df['_team'] == opponent) & 
            (all_teams_df['date'] == match_date) &
            (all_teams_df['opponent'] == team)
        ]
        
        if len(opponent_match) > 0:
            return opponent_match['_team_rest_days'].values[0]
        else:
            # If not found (shouldn't happen for completed matches)
            return 7  # Default to a week 

    def _calculate_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to calculate basic time features needed for advanced features.
        This is used when create_advanced_features is called directly without time series features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with basic time features added
        """
        result_df = df.copy()
        
        # Group by team and create basic features
        new_features = []
        for team in result_df['team'].unique():
            # Get data for this team and sort by date
            team_data = result_df[result_df['team'] == team].sort_values('date')
            
            # Create a separate DataFrame for calculations
            team_calc = team_data.copy()
            
            # Convert result to points: W=3, D=1, L=0
            team_calc['match_points'] = team_calc['result'].map({'W': 3, 'D': 1, 'L': 0})

            # Calculate cumulative sum of points
            team_calc['cumulative_points'] = team_calc['match_points'].cumsum()

            # Shift to ensure we only use past matches (avoid data leakage)
            team_calc['cumulative_points'] = team_calc['cumulative_points'].shift(1)

            # First match should have 0 points
            team_calc['cumulative_points'] = team_calc['cumulative_points'].fillna(0)
            
            new_features.append(team_calc)
            
        # Combine all teams back together
        result_df = pd.concat(new_features)
        
        # Sort back to original order
        result_df = result_df.sort_index()
        
        return result_df

# Add main function for command line use
def main():
    """
    Main function for command line execution.
    Allows users to preprocess data with specified parameters.
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess soccer match data for ML models')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the directory containing the dataset')
    
    # Optional arguments
    parser.add_argument('--league', type=str, default='EPL',
                        help='League name (default: EPL)')
    parser.add_argument('--season', type=str, default='2024',
                        help='Season year (default: 2024)')
    parser.add_argument('--small_window', type=int, default=3,
                        help='Size of the small rolling window (default: 3)')
    parser.add_argument('--window', type=int, default=5,
                        help='Size of the standard rolling window (default: 5)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional path to save preprocessed data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create DataLoader instance
    loader = DataLoader(
        data_path=args.data_path,
        league=args.league,
        season=args.season
    )
    
    print(f"Loading and preprocessing data for {args.league} {args.season}...")
    
    # Preprocess the data
    X, y = loader.preprocess_data()
    
    print(f"\nPreprocessing complete!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # If output file is specified, save the preprocessed data
    if hasattr(args, 'output_file') and args.output_file:
        # Combine features and target into one DataFrame
        output_df = pd.DataFrame(X)
        output_df['result'] = y  # Use 'result' as the default target variable
        
        # Save to file
        output_df.to_csv(args.output_file, index=False)
        print(f"Preprocessed data saved to {args.output_file}")
    
    return X, y

if __name__ == "__main__":
    main() 
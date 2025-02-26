import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

class DataUtils:
    """
    Utility class providing data processing functions for soccer match data.
    """
    
    @staticmethod
    def fill_missing_values(
        df: pd.DataFrame, 
        numerical_features: List[str],
        categorical_features: List[str],
        strategy: str = 'median'
    ) -> pd.DataFrame:
        """
        Fill missing values in the dataset using appropriate strategies for different column types.
        
        Args:
            df (pd.DataFrame): The dataframe containing potentially missing values
            numerical_features (List[str]): List of numerical feature names
            categorical_features (List[str]): List of categorical feature names
            strategy (str): Strategy to use for filling values.
                           'auto': Use appropriate strategy based on column type
                           'mean': Use mean for numerical columns
                           'median': Use median for numerical columns
                           'zero': Use 0 for numerical columns
                           'mode': Use most frequent value for all columns
        
        Returns:
            pd.DataFrame: The dataset with missing values filled
        """
        if df is None or df.empty:
            raise ValueError("No data provided for missing value imputation")
            
        print(f"Filling missing values using strategy: {strategy}")
        
        # Create a copy to avoid modifying the original during processing
        df_copy = df.copy()
        
        # Count missing values before filling
        missing_counts = df_copy.isna().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            print(f"Found {total_missing} missing values across {sum(missing_counts > 0)} columns")
        else:
            print("No missing values found in the dataset")
            return df_copy
            
        # Process numerical features
        for col in numerical_features:
            if col in df_copy.columns and df_copy[col].isna().any():
                num_missing = df_copy[col].isna().sum()
                
                if strategy == 'auto' or strategy == 'mean':
                    fill_value = df_copy[col].mean()
                    method_name = 'mean'
                elif strategy == 'median':
                    fill_value = df_copy[col].median()
                    method_name = 'median'
                elif strategy == 'zero':
                    fill_value = 0
                    method_name = 'zero'
                elif strategy == 'mode':
                    fill_value = df_copy[col].mode()[0]
                    method_name = 'mode'
                else:
                    fill_value = 0
                    method_name = 'zero (default)'
                    
                df_copy[col] = df_copy[col].fillna(fill_value)
                print(f"  - Filled {num_missing} missing values in '{col}' with {method_name}: {fill_value:.4f}")
                
        # Process categorical features
        for col in categorical_features:
            if col in df_copy.columns and df_copy[col].isna().any():
                num_missing = df_copy[col].isna().sum()
                
                if strategy == 'mode':
                    # Use most common value
                    fill_value = df_copy[col].mode()[0]
                    method_name = 'most frequent value'
                else:
                    # Default for categorical is most frequent, with fallback to "Unknown"
                    if df_copy[col].nunique() > 0 and not df_copy[col].mode().empty:
                        fill_value = df_copy[col].mode()[0]
                        method_name = 'most frequent value'
                    else:
                        fill_value = "Unknown"
                        method_name = 'default "Unknown"'
                
                df_copy[col] = df_copy[col].fillna(fill_value)
                print(f"  - Filled {num_missing} missing values in '{col}' with {method_name}: {fill_value}")
        
        # Handle any remaining columns not in our predefined lists
        remaining_cols = [col for col in df_copy.columns 
                        if col not in numerical_features 
                        and col not in categorical_features]
        
        for col in remaining_cols:
            if df_copy[col].isna().any():
                num_missing = df_copy[col].isna().sum()
                
                # Determine if column is numeric or categorical
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    if strategy == 'auto' or strategy == 'mean':
                        fill_value = df_copy[col].mean()
                        method_name = 'mean'
                    elif strategy == 'median':
                        fill_value = df_copy[col].median()
                        method_name = 'median'
                    elif strategy == 'zero':
                        fill_value = 0
                        method_name = 'zero'
                    else:
                        fill_value = 0
                        method_name = 'zero (default)'
                else:
                    # Categorical or other type
                    if df_copy[col].nunique() > 0 and not df_copy[col].mode().empty:
                        fill_value = df_copy[col].mode()[0]
                        method_name = 'most frequent value'
                    else:
                        fill_value = "Unknown"
                        method_name = 'default "Unknown"'
                
                df_copy[col] = df_copy[col].fillna(fill_value)
                print(f"  - Filled {num_missing} missing values in '{col}' with {method_name}")
        
        print(f"Successfully filled all missing values in the dataset")
        return df_copy

    @staticmethod
    def encode_categorical_features(
        df: pd.DataFrame, 
        categorical_features: List[str],
        encoders: Dict = None,
        fit: bool = True
    ) -> tuple:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df (pd.DataFrame): The dataframe containing categorical features
            categorical_features (List[str]): List of categorical feature names
            encoders (Dict, optional): Dictionary of existing encoders
            fit (bool): Whether to fit new encoders or use existing ones
            
        Returns:
            tuple: (encoded_df, encoders) - DataFrame with encoded features and encoders dictionary
        """
        from sklearn.preprocessing import LabelEncoder
        
        df_copy = df.copy()
        encoders = encoders or {}
        
        for feature in categorical_features:
            if feature in df_copy.columns:
                if feature not in encoders:
                    encoders[feature] = LabelEncoder()
                    df_copy[feature] = encoders[feature].fit_transform(df_copy[feature])
                else:
                    # For features where we already have an encoder
                    if fit:
                        df_copy[feature] = encoders[feature].fit_transform(df_copy[feature])
                    else:
                        # Handle unseen categories by setting them to a default value
                        # This prevents the transform from failing on new categories
                        unique_classes = set(encoders[feature].classes_)
                        mask = df_copy[feature].apply(lambda x: x not in unique_classes)
                        if mask.any():
                            default_value = encoders[feature].classes_[0]
                            df_copy.loc[mask, feature] = default_value
                        
                        df_copy[feature] = encoders[feature].transform(df_copy[feature])
        
        return df_copy, encoders 
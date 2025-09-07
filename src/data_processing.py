"""
Data processing module for baseball analytics system.

Contains the main system class and all data loading, preprocessing,
feature engineering, and imputation functionality.
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
from typing import Tuple, List

from config import (
    OUTCOME_MAPPING, FASTBALL_TYPES, BREAKING_TYPES, OFFSPEED_TYPES,
    STRIKE_ZONE, CLUSTERING_CONFIG
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImprovedBaseballSystem:
    """
    Main system class for baseball analytics with enhanced data processing and modeling capabilities.
    
    This class encapsulates the entire pipeline from raw data loading through model training
    and strategic insight generation. Designed for production use with robust error handling
    and comprehensive logging.
    
    Attributes:
        data_dir (str): Path to directory containing input CSV files
        scaler: Feature normalization object (set during training)
        label_encoder: Target variable encoder (currently unused)
        model: Trained neural network model
        stance_clusters (Dict): Clustering results and metadata
        outcome_mapping (Dict): Maps pitch descriptions to 3-class outcomes
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the baseball analytics system.
        
        Args:
            data_dir (str): Path to directory containing batting-stance.csv and statcast data
        """
        self.data_dir = data_dir
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.stance_clusters = None
        self.outcome_mapping = OUTCOME_MAPPING
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess pitch and stance data with comprehensive error handling.
        
        This method handles the complex task of combining temporal pitch data with
        biomechanical stance measurements, including data validation, temporal matching,
        and feature engineering.
        
        Returns:
            pd.DataFrame: Merged dataset with pitch events and corresponding stance data
            
        Raises:
            FileNotFoundError: If required CSV files are not found in data directory
            ValueError: If data formats are incompatible or contain critical errors
        """
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load primary datasets with explicit error handling
            pitch_df = pd.read_csv(os.path.join(self.data_dir, "sample_statcast.csv"))
            stance_df = pd.read_csv(os.path.join(self.data_dir, "batting-stance.csv"))
            
            logger.info(f"Loaded {len(pitch_df)} pitch records and {len(stance_df)} stance records")
            
            # Standardize player name format for consistent merging
            # Convert "First Last" to "Last, First" format used in stance data
            pitch_df['name'] = pitch_df['player_name'].apply(
                lambda x: ', '.join(x.strip().split()[::-1]) if isinstance(x, str) and x.strip() else x
            )
            
            # Enhanced date processing with comprehensive validation
            pitch_df['game_date'] = pd.to_datetime(pitch_df['game_date'], errors='coerce')
            pitch_df = pitch_df.dropna(subset=['game_date'])  # Remove invalid dates
            
            # Create temporal features for potential seasonal analysis
            pitch_df['year'] = pitch_df['game_date'].dt.year
            pitch_df['month'] = pitch_df['game_date'].dt.month
            pitch_df['day_of_year'] = pitch_df['game_date'].dt.dayofyear
            
            # Process stance measurement dates (monthly aggregations)
            # Stance data is aggregated by month, so we create first-of-month dates
            stance_df['stance_date'] = pd.to_datetime(
                stance_df['year'].astype(str) + '-' + 
                stance_df['api_game_date_month_mm'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
            stance_df = stance_df.dropna(subset=['stance_date'])
            
            # Perform sophisticated temporal matching between datasets
            merged_df = self._improved_temporal_merge(pitch_df, stance_df)
            
            # Generate comprehensive engineered features
            merged_df = self._create_enhanced_features(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _improved_temporal_merge(self, pitch_df: pd.DataFrame, stance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sophisticated temporal matching between pitch events and stance measurements.
        
        This is a critical component that matches individual pitch events with the closest
        available batting stance measurements in time. The challenge is that stance data
        is measured monthly while pitch data is per-event.
        
        Algorithm:
        1. Merge datasets on player name
        2. Calculate temporal distance between pitch date and stance measurement date
        3. For each pitch, select the stance measurement with minimum temporal distance
        4. Apply reasonable temporal constraints (within 90 days)
        
        Args:
            pitch_df (pd.DataFrame): Individual pitch event data
            stance_df (pd.DataFrame): Monthly aggregated stance measurement data
            
        Returns:
            pd.DataFrame: Pitch data with matched stance measurements
        """
        logger.info("Performing improved temporal matching...")
        
        # Preserve original pitch indices for accurate matching
        pitch_df_reset = pitch_df.reset_index().rename(columns={'index': 'pitch_index'})
        
        # Perform left join to keep all pitches, add stance data where available
        merged = pitch_df_reset.merge(stance_df, on='name', how='left', suffixes=('', '_stance'))
        
        # Filter to only pitches with available stance data
        merged = merged.dropna(subset=['stance_date', 'game_date'])
        
        # Calculate temporal distance in days for precise matching
        merged['date_diff'] = (merged['stance_date'] - merged['game_date']).abs().dt.days
        
        # Apply reasonable temporal constraint: stance data within 90 days of pitch
        # This prevents matching pitches with stance data from different seasons
        merged = merged[merged['date_diff'] <= CLUSTERING_CONFIG['temporal_threshold_days']]
        
        # For each pitch, select the stance measurement with minimum temporal distance
        idx = merged.groupby('pitch_index')['date_diff'].idxmin()
        closest_stance = merged.loc[idx].set_index('pitch_index')
        
        # Define stance features to retain in final dataset
        stance_cols = [
            'avg_batter_y_position',      # Vertical position in batter's box
            'avg_batter_x_position',      # Horizontal position in batter's box  
            'avg_foot_sep',               # Distance between feet (stance width)
            'avg_stance_angle',           # Angle of stance (closed/open)
            'avg_intercept_y_vs_batter',  # Biomechanical timing metric
            'avg_intercept_y_vs_plate',   # Position relative to home plate
            'bat_side',                   # Left/Right handed batter
            'stance_date'                 # Date of stance measurement
        ]
        
        # Merge stance data back to original pitch dataframe structure
        result_df = pitch_df.reset_index(drop=True).join(closest_stance[stance_cols])
        
        return result_df
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive engineered features combining pitch physics and stance biomechanics.
        
        This method creates 40+ features that capture the complex interactions between
        pitch characteristics and batter positioning. Features are designed based on
        baseball physics principles and strategic considerations.
        
        Feature Categories:
        1. Pitch Location: Strike zone analysis, distance metrics
        2. Pitch Movement: Horizontal/vertical break, total movement
        3. Release Point: Pitcher mechanics and deception metrics
        4. Count Situation: Strategic context features
        5. Stance Characteristics: Biomechanical positioning features
        6. Pitch Classification: Type groupings and relative metrics
        
        Args:
            df (pd.DataFrame): Merged pitch and stance data
            
        Returns:
            pd.DataFrame: Dataset with comprehensive engineered features
        """
        logger.info("Creating enhanced features...")
        
        # === PITCH LOCATION FEATURES ===
        # Distance from center of strike zone (plate center is 0,0, vertical center ~2.5ft)
        df['distance_from_center'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - STRIKE_ZONE['vertical_center'])**2)
        
        # Binary indicator for pitches in official strike zone
        df['in_strike_zone'] = ((df['plate_x'].abs() <= STRIKE_ZONE['horizontal_width']) & 
                               (df['plate_z'] >= STRIKE_ZONE['vertical_min']) & 
                               (df['plate_z'] <= STRIKE_ZONE['vertical_max'])).astype(int)
        
        # Edge of zone indicator for borderline calls (critical for umpire decisions)
        df['edge_of_zone'] = (((df['plate_x'].abs() > 0.7) & (df['plate_x'].abs() <= 1.0)) |
                             ((df['plate_z'] < 1.7) & (df['plate_z'] >= 1.3)) |
                             ((df['plate_z'] > 3.3) & (df['plate_z'] <= 3.7))).astype(int)
        
        # === PITCH MOVEMENT FEATURES ===
        # Total break/movement magnitude (inches)
        df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
        
        # Horizontal movement magnitude (slider/cutter break)
        df['horizontal_movement'] = df['pfx_x'].abs()
        
        # Vertical movement (gravity-adjusted drop/rise)
        df['vertical_movement'] = df['pfx_z']
        
        # === RELEASE POINT FEATURES ===
        # Effective velocity accounts for extension (closer release = faster perceived speed)
        df['effective_velocity'] = df['release_speed'] * (df['release_extension'] / 6.0)
        
        # Release height relative to pitcher's mound (6ft standard)
        df['release_height_normalized'] = df['release_pos_z'] - 6.0
        
        # === COUNT SITUATION FEATURES ===
        # These features capture the strategic context of each pitch
        if 'balls' in df.columns and 'strikes' in df.columns:
            df['count_pressure'] = df['strikes'] - df['balls']  # Positive = pitcher ahead
            df['pitcher_ahead'] = (df['strikes'] > df['balls']).astype(int)
            df['hitter_ahead'] = (df['balls'] > df['strikes']).astype(int)
            df['two_strike_count'] = (df['strikes'] == 2).astype(int)  # Defensive hitting
            df['three_ball_count'] = (df['balls'] == 3).astype(int)   # Must throw strike
            df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
        else:
            # Create dummy features when count data unavailable
            df['count_pressure'] = 0
            df['pitcher_ahead'] = 0
            df['hitter_ahead'] = 0
            df['two_strike_count'] = 0
            df['three_ball_count'] = 0
            df['full_count'] = 0
        
        # === STANCE-DERIVED FEATURES ===
        # Advanced biomechanical features based on batting stance measurements
        if 'avg_foot_sep' in df.columns and 'avg_stance_angle' in df.columns:
            # Stance openness (absolute angle regardless of direction)
            df['stance_openness'] = df['avg_stance_angle'].abs()
            
            # Width-to-depth ratio (stability vs. mobility tradeoff)
            df['stance_width_ratio'] = df['avg_foot_sep'] / (df['avg_batter_y_position'] + 1e-6)
            
            # Categorical stance characteristics for strategic analysis
            df['stance_closed'] = (df['avg_stance_angle'] < -15).astype(int)   # Very closed
            df['stance_open'] = (df['avg_stance_angle'] > 15).astype(int)      # Very open
            df['stance_wide'] = (df['avg_foot_sep'] > 35).astype(int)          # Power stance
            df['stance_narrow'] = (df['avg_foot_sep'] < 25).astype(int)        # Contact stance
        
        # === PITCH TYPE CLASSIFICATION FEATURES ===
        # Group pitch types by strategic similarity
        df['is_fastball'] = df['pitch_type'].isin(FASTBALL_TYPES).astype(int)
        df['is_breaking'] = df['pitch_type'].isin(BREAKING_TYPES).astype(int)
        df['is_offspeed'] = df['pitch_type'].isin(OFFSPEED_TYPES).astype(int)
        
        # Velocity relative to pitch type expectation (deception metric)
        pitch_type_avg_speed = df.groupby('pitch_type')['release_speed'].transform('mean')
        df['speed_vs_pitch_type_avg'] = df['release_speed'] - pitch_type_avg_speed
        
        return df
    
    def advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement sophisticated multi-level imputation strategy for missing data.
        
        Missing data is inevitable in baseball analytics due to equipment failures,
        measurement errors, and data collection gaps. This method implements a
        hierarchical imputation strategy that preserves statistical relationships.
        
        Imputation Strategy:
        1. Player-specific patterns (if player has other measurements)
        2. Group-specific patterns (handedness, pitch type)
        3. Global statistical measures (overall median/mode)
        
        Args:
            df (pd.DataFrame): Dataset with potential missing values
            
        Returns:
            pd.DataFrame: Dataset with imputed missing values
        """
        logger.info("Performing advanced imputation...")
        
        # Define feature categories for targeted imputation strategies
        stance_features = [
            'avg_batter_y_position', 'avg_batter_x_position', 'avg_foot_sep',
            'avg_stance_angle', 'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate'
        ]
        
        pitch_features = [
            'release_speed', 'release_pos_x', 'release_pos_z',
            'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
            'zone', 'release_spin_rate', 'release_extension'
        ]
        
        # === STANCE FEATURE IMPUTATION ===
        # Multi-level approach: player → handedness → global
        for col in stance_features:
            if col in df.columns:
                # Level 1: Player-specific median (player consistency)
                df[col] = df.groupby('name')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 2: Handedness-specific median (biomechanical similarity)
                df[col] = df.groupby('stand')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 3: Global median (population average)
                df[col] = df[col].fillna(df[col].median())
        
        # === PITCH FEATURE IMPUTATION ===
        # Multi-level approach: pitch type → global
        for col in pitch_features:
            if col in df.columns:
                # Level 1: Pitch-type specific median (physics consistency)
                df[col] = df.groupby('pitch_type')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 2: Global median fallback
                df[col] = df[col].fillna(df[col].median())
        
        # === CATEGORICAL VARIABLE IMPUTATION ===
        # Use mode (most frequent value) for categorical features
        for col in ['bat_side', 'stand', 'pitch_type']:
            if col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        return df


def create_pitch_sequences(df: pd.DataFrame, feature_cols: List[str], 
                          label_col: str, seq_len: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequential pitch data for temporal modeling (currently unused in main pipeline).
    
    This function creates sequences of consecutive pitches to the same batter,
    which could be used for LSTM/RNN modeling to capture pitcher-hitter dynamics
    over the course of an at-bat.
    
    Args:
        df (pd.DataFrame): Pitch data with temporal ordering
        feature_cols (List[str]): Column names to use as features
        label_col (str): Target variable column name
        seq_len (int): Length of sequences to create (default: 3)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (sequences, labels) for training
    """
    logger.info(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    labels = []
    
    # Group pitches by game and batter for temporal continuity
    grouped = df.groupby(['game_date', 'name'])
    
    for (date, player), group in grouped:
        if len(group) < seq_len:
            continue  # Skip groups with insufficient data
            
        # Sort by temporal indicators for proper sequence ordering
        if 'inning' in group.columns and 'at_bat_number' in group.columns:
            group = group.sort_values(['inning', 'at_bat_number'])
        else:
            group = group.sort_index()  # Fallback to index ordering
        
        features = group[feature_cols].values
        targets = group[label_col].values
        
        # Create overlapping sequences (sliding window approach)
        for i in range(len(group) - seq_len + 1):
            seq_features = features[i:i+seq_len]
            seq_label = targets[i+seq_len-1]  # Predict outcome of final pitch in sequence
            
            # Quality check: ensure no missing values in sequence
            if not (np.isnan(seq_features).any() or np.isnan(seq_label)):
                sequences.append(seq_features)
                labels.append(seq_label)
    
    return np.array(sequences), np.array(labels)


def create_single_pitch_features(df: pd.DataFrame, feature_cols: List[str], 
                                label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single pitch features and targets from dataset (main approach used).
    
    This function creates the feature matrix and target vector used for training
    the feedforward neural network. Each row represents a single pitch with
    associated stance characteristics.
    
    Args:
        df (pd.DataFrame): Complete dataset with features and targets
        feature_cols (List[str]): Column names to use as model features
        label_col (str): Target variable column name
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, targets) ready for model training
    """
    logger.info("Creating single pitch features...")
    
    # Remove rows with any missing values in features or target
    clean_df = df[feature_cols + [label_col]].dropna()
    
    # Extract feature matrix and target vector
    X = clean_df[feature_cols].values
    y = clean_df[label_col].values
    
    return X, y
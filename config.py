"""
Configuration file for baseball analytics system.

Contains all constants, mappings, and configuration parameters used throughout the project.
"""

# Outcome mapping strategy: Convert descriptive pitch outcomes to strategic categories
# 0 = Pitcher Win (strikes, swinging strikes)
# 1 = Neutral (fouls, blocked balls, pitchouts - extend at-bat)
# 2 = Hitter Win (balls, HBP, contact opportunities)
OUTCOME_MAPPING = {
    'swinging_strike': 0,        # Clear pitcher advantage
    'swinging_strike_blocked': 0,
    'called_strike': 0,
    'missed_bunt': 0,
    'pitchout': 1,               # Strategic neutral outcome
    'blocked_ball': 1,           # Prevents advancement but no strike
    'foul': 1,                   # Extends at-bat, neutral value
    'bunt_foul_tip': 1,
    'foul_bunt': 1,
    'foul_tip': 0,               # Counts as strike
    'ball': 2,                   # Hitter advantage (improves count)
    'hit_by_pitch': 2,           # Free base for hitter
    'hit_into_play': 2           # Contact opportunity for hitter
}

# Feature columns for model training
FEATURE_COLS = [
    # Basic pitch features
    'release_speed', 'release_pos_x', 'release_pos_z',
    'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
    'zone', 'release_spin_rate', 'release_extension',
    
    # Enhanced features
    'distance_from_center', 'in_strike_zone', 'edge_of_zone',
    'total_movement', 'horizontal_movement', 'vertical_movement',
    'effective_velocity', 'release_height_normalized',
    
    # Count features
    'count_pressure', 'pitcher_ahead', 'hitter_ahead',
    'two_strike_count', 'three_ball_count', 'full_count',
    
    # Stance features
    'avg_batter_y_position', 'avg_batter_x_position',
    'avg_foot_sep', 'avg_stance_angle',
    'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate',
    'stance_openness', 'stance_width_ratio',
    'stance_closed', 'stance_open', 'stance_wide', 'stance_narrow',
    
    # Pitch type features
    'is_fastball', 'is_breaking', 'is_offspeed',
    'speed_vs_pitch_type_avg'
]

# Stance features for clustering
STANCE_FEATURES = [
    'avg_batter_y_position',      # Depth in batter's box
    'avg_batter_x_position',      # Distance from plate
    'avg_foot_sep',               # Stance width
    'avg_stance_angle',           # Open/closed orientation
    'avg_intercept_y_vs_batter',  # Biomechanical timing
    'avg_intercept_y_vs_plate',   # Plate coverage metric
    'bat_side_enc'                # Handedness (numeric)
]

# Pitch type classifications
FASTBALL_TYPES = ['FF', 'FA', 'FT', 'FC', 'SI']  # Velocity-based pitches
BREAKING_TYPES = ['SL', 'CU', 'KC', 'SV', 'ST']  # Sharp break pitches
OFFSPEED_TYPES = ['CH', 'FS', 'FO', 'SC']        # Change of pace pitches

# Model hyperparameters
MODEL_CONFIG = {
    'hidden_size': 128,
    'num_classes': 3,
    'dropout': 0.4,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 1024,
    'patience': 15
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'n_clusters': 5,
    'random_state': 42,
    'n_init': 10,
    'temporal_threshold_days': 90  # Max days between pitch and stance data
}

# File paths
DATA_DIR = "data"
RESULTS_DIR = "results"

# Strike zone parameters (for feature engineering)
STRIKE_ZONE = {
    'horizontal_width': 0.83,  # Half-width of strike zone in feet
    'vertical_min': 1.5,       # Bottom of strike zone in feet
    'vertical_max': 3.5,       # Top of strike zone in feet
    'vertical_center': 2.5     # Center of strike zone in feet
}

# Target names for classification report
TARGET_NAMES = ['Pitcher Win', 'Neutral', 'Hitter Win']

# Minimum sample sizes for analysis
MIN_SAMPLES = {
    'pitch_type_analysis': 50,    # Min pitches per type for stance analysis
    'player_recommendation': 10   # Min pitches for player-specific advice
}
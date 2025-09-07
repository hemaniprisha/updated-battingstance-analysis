"""
Main execution file for the Baseball Analytics System.

This file orchestrates the complete pipeline from data loading through 
model training, evaluation, and insight generation.

Author: Prisha Hemani
Version: 2.0
Last Updated: 07/09/2025
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from collections import Counter
import logging
import warnings

# Import our modules
from src.data_processing import ImprovedBaseballSystem, create_single_pitch_features
from src.models import train_improved_model, evaluate_classification_model
from src.analysis import perform_stance_clustering, analyze_stance_performance, generate_recommendations
from src.visualization import plot_results, export_results, save_summary_report
from config import OUTCOME_MAPPING, FEATURE_COLS, DATA_DIR

# Configure logging and suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the end-to-end pipeline: data prep, clustering, training, evaluation, insights.

    Steps:
        1) Load raw pitch and stance data, engineer features, and impute missing values.
        2) Map raw Statcast descriptions to 3-class outcome categories.
        3) Perform K-means stance clustering to discover stance archetypes.
        4) Build train/val/test splits with stratification and scale features.
        5) Train the neural network with class-balancing and early stopping.
        6) Evaluate on the held-out test set and log metrics.
        7) Analyze stance-cluster performance and generate recommendations.

    Returns:
        Tuple: (system, model, results, df, stance_analysis)
            - system: ImprovedBaseballSystem instance with clustering metadata.
            - model: Trained ImprovedPitchClassifier.
            - results: Dict with evaluation metrics and predictions.
            - df: Processed DataFrame used for modeling and analysis.
            - stance_analysis: Per-cluster performance summary DataFrame.
    """
    logger.info("Starting baseball analytics system...")
    
    try:
        # Initialize system
        system = ImprovedBaseballSystem(data_dir=DATA_DIR)
        
        # Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        df = system.load_and_preprocess_data()
        df = system.advanced_imputation(df)
        
        # Map outcomes to categories
        logger.info("Step 2: Mapping pitch outcomes to categories...")
        df['outcome_category'] = df['description'].map(OUTCOME_MAPPING)
        df = df.dropna(subset=['outcome_category'])
        
        logger.info(f"Outcome distribution: {Counter(df['outcome_category'])}")
        
        # Perform stance clustering
        logger.info("Step 3: Performing stance clustering...")
        cluster_stats = perform_stance_clustering(df)
        system.stance_clusters = cluster_stats  # Store in system for later use
        
        # Define features for model
        available_features = [col for col in FEATURE_COLS if col in df.columns]
        logger.info(f"Using {len(available_features)} features out of {len(FEATURE_COLS)} total")
        
        # Create features (single pitch, not sequences)
        logger.info("Step 4: Creating feature matrices...")
        X, y = create_single_pitch_features(df, available_features, 'outcome_category')
        logger.info(f"Created {X.shape[0]} samples with {X.shape[1]} features")
        
        # Stratified split to maintain class balance
        logger.info("Step 5: Creating train/validation/test splits...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        
        X_temp, X_test = X[train_idx], X[test_idx]
        y_temp, y_test = y[train_idx], y[test_idx]
        
        # Further split for validation
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss_val.split(X_temp, y_temp))
        
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        # Scale features
        logger.info("Step 6: Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        system.scaler = scaler  # Store scaler in system
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Train model
        logger.info("Step 7: Training neural network model...")
        model, train_losses, val_losses = train_improved_model(
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
            input_size=X_train_scaled.shape[1], num_classes=3
        )
        
        system.model = model  # Store model in system
        
        # Evaluate model
        logger.info("Step 8: Evaluating model performance...")
        results = evaluate_classification_model(model, X_test_tensor, y_test_tensor)
        
        # Analyze stance effectiveness
        logger.info("Step 9: Analyzing stance cluster performance...")
        stance_analysis = analyze_stance_performance(df)
        
        if not stance_analysis.empty:
            logger.info("\nStance Analysis Results:")
            print(stance_analysis.round(4).to_string(index=False))
        
        # Print cluster insights
        if cluster_stats:
            logger.info("\nCluster Insights:")
            for cluster_id, stats in cluster_stats.get('stats', {}).items():
                logger.info(f"\nCluster {cluster_id}: {stats['size']} pitches")
                logger.info(f"  Avg foot separation: {stats['means'].get('avg_foot_sep', 'N/A'):.2f}")
                logger.info(f"  Avg stance angle: {stats['means'].get('avg_stance_angle', 'N/A'):.2f}°")
                logger.info(f"  Handedness mix: {stats['means'].get('bat_side_enc', 'N/A'):.2f}")
        
        # Generate recommendations
        logger.info("Step 10: Generating actionable recommendations...")
        recommendations = generate_recommendations(stance_analysis)
        logger.info("\nActionable Recommendations:")
        for category, recs in recommendations.items():
            logger.info(f"\n{category}:")
            for rec in recs:
                logger.info(f"  • {rec}")
        
        # Create visualizations
        logger.info("Step 11: Creating visualizations...")
        plot_results(train_losses, val_losses, results, stance_analysis)
        
        # Export comprehensive results
        logger.info("Step 12: Exporting results...")
        try:
            export_results(results, stance_analysis, cluster_stats.get('stats', {}))
        except Exception as e:
            logger.warning(f"JSON export failed: {str(e)}")
            logger.info("Continuing without JSON export...")
        
        save_summary_report(results, stance_analysis, recommendations)
        
        logger.info("Baseball analytics pipeline completed successfully!")
        
        return system, model, results, df, stance_analysis
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


def run_player_analysis(player_name: str, system=None, df=None, stance_analysis=None):
    """Run analysis for a specific player.
    
    Args:
        player_name (str): Player name in "Last, First" format
        system: Trained system instance (optional, will run main if not provided)
        df: Processed dataframe (optional)
        stance_analysis: Stance analysis results (optional)
    """
    from src.analysis import get_player_stance_recommendation
    
    if system is None or df is None or stance_analysis is None:
        logger.info("Running full analysis first...")
        system, _, _, df, stance_analysis = main()
    
    recommendation = get_player_stance_recommendation(player_name, df, stance_analysis)
    print(f"\n{recommendation}")


if __name__ == "__main__":
    # Run the complete analysis pipeline
    system, model, results, data, stance_analysis = main()
    
    # Optional: Run player-specific analysis
    # Uncomment and modify the player name as needed
    # run_player_analysis("Olson, Matt", system, data, stance_analysis)
    
    print("\nAnalysis complete! Check the generated files:")
    print("- baseball_analysis_results.png (visualization)")
    print("- baseball_analysis_results.json (detailed results)")
    print("- analysis_summary.txt (summary report)")
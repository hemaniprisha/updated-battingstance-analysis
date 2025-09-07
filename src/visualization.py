"""
Visualization and export functionality for baseball analytics results.

Contains functions for creating plots, dashboards, and exporting results to various formats.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
from sklearn.metrics import confusion_matrix
from typing import List, Dict

from config import TARGET_NAMES

logger = logging.getLogger(__name__)


def plot_results(train_losses: List[float], val_losses: List[float], 
                results: Dict, stance_analysis: pd.DataFrame):
    """Create a 2x2 dashboard of training curves, confusion matrix, and stance insights.

    Args:
        train_losses (List[float]): Per-epoch training loss values.
        val_losses (List[float]): Per-epoch validation loss values.
        results (Dict): Output from evaluate_classification_model containing
            predictions and actuals for confusion matrix.
        stance_analysis (pd.DataFrame): Aggregated stance performance metrics for
            bar and scatter plots.

    Notes:
        - Saves a high-resolution image to 'baseball_analysis_results.png'.
        - Displays the figure for interactive review.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(results['actuals'], results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], 
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES)
    axes[0, 1].set_title('Confusion Matrix')
    
    # Stance cluster performance
    if not stance_analysis.empty:
        x_pos = range(len(stance_analysis))
        axes[1, 0].bar(x_pos, stance_analysis['pitcher_win_rate'], 
                      color='lightcoral', alpha=0.7)
        axes[1, 0].set_xlabel('Stance Cluster')
        axes[1, 0].set_ylabel('Pitcher Win Rate')
        axes[1, 0].set_title('Pitcher Success by Stance Cluster')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f"Cluster {int(c)}" for c in stance_analysis['stance_cluster']])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stance characteristics
        scatter = axes[1, 1].scatter(stance_analysis['avg_stance_angle'], 
                                   stance_analysis['avg_foot_separation'],
                                   c=stance_analysis['pitcher_win_rate'],
                                   s=stance_analysis['total_pitches']/100,
                                   alpha=0.7, cmap='RdYlBu_r')
        axes[1, 1].set_xlabel('Avg Stance Angle (degrees)')
        axes[1, 1].set_ylabel('Avg Foot Separation')
        axes[1, 1].set_title('Stance Characteristics vs Performance')
        plt.colorbar(scatter, ax=axes[1, 1], label='Pitcher Win Rate')
        
        # Add cluster labels
        for _, row in stance_analysis.iterrows():
            axes[1, 1].annotate(f"C{int(row['stance_cluster'])}", 
                              (row['avg_stance_angle'], row['avg_foot_separation']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('baseball_analysis_results.png', dpi=300, bbox_inches='tight')
    logger.info("Results visualization saved as 'baseball_analysis_results.png'")
    plt.show()


def export_results(results: Dict, stance_analysis: pd.DataFrame, 
                  cluster_stats: Dict, filename: str = "baseball_analysis_results.json"):
    """Export comprehensive results to JSON format.
    
    Args:
        results (Dict): Model evaluation results from evaluate_classification_model
        stance_analysis (pd.DataFrame): Aggregated stance cluster performance metrics
        cluster_stats (Dict): Detailed clustering statistics and metadata
        filename (str): Output filename for JSON export
    """
    
    export_data = {
        "model_performance": {
            "accuracy": float(results['accuracy']),
            "f1_weighted": float(results['f1_weighted']),
            "classification_report": results['classification_report']
        },
        "stance_analysis": stance_analysis.to_dict('records') if not stance_analysis.empty else [],
        "cluster_statistics": cluster_stats,
        "summary": {
            "total_clusters": len(cluster_stats),
            "best_performing_cluster": int(stance_analysis.loc[stance_analysis['pitcher_win_rate'].idxmax(), 'stance_cluster']) if not stance_analysis.empty else None,
            "performance_range": {
                "min_pitcher_success": float(stance_analysis['pitcher_win_rate'].min()) if not stance_analysis.empty else None,
                "max_pitcher_success": float(stance_analysis['pitcher_win_rate'].max()) if not stance_analysis.empty else None
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Results exported to {filename}")


def create_stance_cluster_plot(stance_analysis: pd.DataFrame):
    """Create detailed visualization of stance cluster characteristics.
    
    Args:
        stance_analysis (pd.DataFrame): Stance cluster performance data
    """
    if stance_analysis.empty:
        logger.warning("No stance analysis data to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Performance comparison
    axes[0].bar(range(len(stance_analysis)), stance_analysis['pitcher_win_rate'])
    axes[0].set_xlabel('Stance Cluster')
    axes[0].set_ylabel('Pitcher Win Rate')
    axes[0].set_title('Pitcher Success Rate by Cluster')
    axes[0].set_xticks(range(len(stance_analysis)))
    axes[0].set_xticklabels([f"C{int(c)}" for c in stance_analysis['stance_cluster']])
    
    # Stance geometry
    axes[1].scatter(stance_analysis['avg_stance_angle'], 
                   stance_analysis['avg_foot_separation'],
                   c=stance_analysis['pitcher_win_rate'],
                   s=100, alpha=0.7, cmap='RdYlBu_r')
    axes[1].set_xlabel('Stance Angle (degrees)')
    axes[1].set_ylabel('Foot Separation')
    axes[1].set_title('Stance Geometry')
    
    # Sample sizes
    axes[2].bar(range(len(stance_analysis)), stance_analysis['total_pitches'])
    axes[2].set_xlabel('Stance Cluster')
    axes[2].set_ylabel('Total Pitches')
    axes[2].set_title('Sample Size by Cluster')
    axes[2].set_xticks(range(len(stance_analysis)))
    axes[2].set_xticklabels([f"C{int(c)}" for c in stance_analysis['stance_cluster']])
    
    plt.tight_layout()
    plt.savefig('stance_cluster_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Stance cluster visualization saved as 'stance_cluster_analysis.png'")
    plt.show()


def create_performance_heatmap(df: pd.DataFrame):
    """Create heatmap showing performance across pitch types and stance clusters.
    
    Args:
        df (pd.DataFrame): Full dataset with stance clusters and pitch outcomes
    """
    if 'stance_cluster' not in df.columns:
        logger.warning("No stance cluster data for heatmap")
        return
    
    # Create pivot table of pitch type vs stance cluster performance
    pivot_data = df.groupby(['pitch_type', 'stance_cluster'])['outcome_category'].mean().unstack()
    
    if pivot_data.empty:
        logger.warning("Insufficient data for performance heatmap")
        return
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=1.0)
    plt.title('Pitch Outcome by Type and Stance Cluster\n(0=Pitcher Win, 1=Neutral, 2=Hitter Win)')
    plt.xlabel('Stance Cluster')
    plt.ylabel('Pitch Type')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    logger.info("Performance heatmap saved as 'performance_heatmap.png'")
    plt.show()


def save_summary_report(results: Dict, stance_analysis: pd.DataFrame, 
                       recommendations: Dict, filename: str = "analysis_summary.txt"):
    """Generate and save a comprehensive text summary report.
    
    Args:
        results (Dict): Model evaluation results
        stance_analysis (pd.DataFrame): Stance cluster performance data
        recommendations (Dict): Generated recommendations by category
        filename (str): Output text file name
    """
    
    with open(filename, 'w') as f:
        f.write("BASEBALL ANALYTICS SYSTEM - ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.3f}\n")
        f.write(f"Weighted F1 Score: {results['f1_weighted']:.3f}\n\n")
        
        # Stance Analysis
        if not stance_analysis.empty:
            f.write("STANCE CLUSTER ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Number of Clusters: {len(stance_analysis)}\n")
            f.write(f"Pitcher Win Rate Range: {stance_analysis['pitcher_win_rate'].min():.3f} - {stance_analysis['pitcher_win_rate'].max():.3f}\n\n")
            
            f.write("Cluster Details:\n")
            for _, row in stance_analysis.iterrows():
                f.write(f"  Cluster {int(row['stance_cluster'])}: ")
                f.write(f"{row['pitcher_win_rate']:.1%} pitcher success, ")
                f.write(f"{row['total_pitches']} pitches, ")
                f.write(f"stance: {row['avg_stance_angle']:.1f}° angle, {row['avg_foot_separation']:.1f} foot sep\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        for category, recs in recommendations.items():
            f.write(f"\n{category}:\n")
            for rec in recs:
                f.write(f"  • {rec}\n")
    
    logger.info(f"Summary report saved as '{filename}'")
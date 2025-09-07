"""
Analysis module for stance clustering and performance evaluation.

Contains functions for clustering batting stances, analyzing performance metrics,
and generating actionable recommendations for players and coaches.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from collections import Counter
import logging
from typing import Dict, List

from config import STANCE_FEATURES, CLUSTERING_CONFIG, MIN_SAMPLES

logger = logging.getLogger(__name__)


def perform_stance_clustering(df: pd.DataFrame, 
                             n_clusters: int = CLUSTERING_CONFIG['n_clusters']) -> Dict:
    """
    Identify distinct batting stance archetypes using unsupervised K-means clustering.
    
    This method discovers natural groupings in batting stance characteristics,
    enabling strategic analysis of stance-based vulnerabilities and strengths.
    The number of clusters (5) was chosen based on baseball domain knowledge
    and statistical validation.
    
    Clustering Features:
    - Batter positioning (x, y coordinates)
    - Stance geometry (foot separation, angle)
    - Biomechanical timing (intercept metrics)
    - Handedness encoding
    
    Args:
        df (pd.DataFrame): Dataset with stance measurements
        n_clusters (int): Number of stance archetypes to identify (default: 5)
        
    Returns:
        Dict: Clustering results including model, statistics, and metadata
    """
    logger.info(f"Performing stance clustering with {n_clusters} clusters...")
    
    # Encode categorical handedness for clustering
    df['bat_side_enc'] = df['bat_side'].map({'R': 0, 'L': 1}).fillna(0.5)
    
    # Extract clean data for clustering (no missing values)
    stance_data = df[STANCE_FEATURES].dropna()
    
    if len(stance_data) == 0:
        logger.warning("No valid stance data for clustering")
        return {}
    
    # Use RobustScaler to handle outliers in biomechanical measurements
    scaler = RobustScaler()
    stance_scaled = scaler.fit_transform(stance_data)
    
    # Perform K-means clustering with multiple random initializations
    kmeans = KMeans(n_clusters=n_clusters, 
                   random_state=CLUSTERING_CONFIG['random_state'], 
                   n_init=CLUSTERING_CONFIG['n_init'])
    clusters = kmeans.fit_predict(stance_scaled)
    
    # Add cluster assignments back to original dataframe
    df['stance_cluster'] = np.nan
    df.loc[stance_data.index, 'stance_cluster'] = clusters
    
    # Calculate comprehensive statistics for each cluster
    cluster_stats = {}
    for i in range(n_clusters):
        mask = df['stance_cluster'] == i
        if mask.sum() > 0:
            cluster_data = df[mask][STANCE_FEATURES]
            cluster_stats[i] = {
                'size': mask.sum(),                    # Number of pitches in cluster
                'means': cluster_data.mean().to_dict(), # Average characteristics
                'stds': cluster_data.std().to_dict()    # Variability measures
            }
    
    return {
        'model': kmeans,           # Fitted K-means model
        'scaler': scaler,          # Feature scaling parameters
        'features': STANCE_FEATURES, # Features used for clustering
        'stats': cluster_stats     # Statistical summaries
    }


def analyze_stance_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stance-cluster performance metrics and pitch-type insights.

    This aggregates pitch outcomes within each discovered stance cluster and
    derives summary statistics useful for strategy (e.g., pitcher win rate,
    hitter win rate), as well as the most and least effective pitch types
    against each stance (subject to a minimum sample size threshold).

    Args:
        df (pd.DataFrame): Dataset containing stance_cluster assignments and
            outcome_category labels.

    Returns:
        pd.DataFrame: One row per stance cluster with aggregated metrics such as
            total_pitches, pitcher_win_rate, neutral_rate, hitter_win_rate,
            average stance characteristics, and best/worst pitch types when
            available.
    """
    if 'stance_cluster' not in df.columns:
        logger.warning("No stance clusters found")
        return pd.DataFrame()
    
    analysis_results = []
    
    for cluster in sorted(df['stance_cluster'].dropna().unique()):
        cluster_data = df[df['stance_cluster'] == cluster]
        
        result = {
            'stance_cluster': int(cluster),
            'total_pitches': len(cluster_data),
            'pitcher_win_rate': (cluster_data['outcome_category'] == 0).mean(),
            'neutral_rate': (cluster_data['outcome_category'] == 1).mean(),
            'hitter_win_rate': (cluster_data['outcome_category'] == 2).mean(),
            'avg_foot_separation': cluster_data['avg_foot_sep'].mean(),
            'avg_stance_angle': cluster_data['avg_stance_angle'].mean(),
            'avg_batter_position': cluster_data['avg_batter_x_position'].mean()
        }
        
        # Best and worst pitch types
        pitch_performance = cluster_data.groupby('pitch_type')['outcome_category'].agg(['mean', 'count'])
        pitch_performance = pitch_performance[pitch_performance['count'] >= MIN_SAMPLES['pitch_type_analysis']]
        
        if not pitch_performance.empty:
            best_pitch = pitch_performance['mean'].idxmin()  # Lowest score = best for pitcher
            worst_pitch = pitch_performance['mean'].idxmax()  # Highest score = best for hitter
            
            result['best_pitch_vs_stance'] = best_pitch
            result['worst_pitch_vs_stance'] = worst_pitch
            result[f'best_pitch_success_rate'] = 1 - pitch_performance.loc[best_pitch, 'mean']  # Invert for pitcher success
            result[f'worst_pitch_success_rate'] = 1 - pitch_performance.loc[worst_pitch, 'mean']
        
        analysis_results.append(result)
    
    return pd.DataFrame(analysis_results)


def generate_recommendations(stance_analysis: pd.DataFrame) -> Dict[str, List[str]]:
    """Translate stance cluster metrics into concise, actionable guidance.

    Args:
        stance_analysis (pd.DataFrame): Output of analyze_stance_performance with
            per-cluster success rates and optional best/worst pitch types.

    Returns:
        Dict[str, List[str]]: Buckets of recommendations for pitching strategy,
            hitting adjustments, and coaching insights.
    """
    
    if stance_analysis.empty:
        return {"No Analysis": ["Insufficient stance data for recommendations"]}
    
    recommendations = {
        "Pitching Strategy": [],
        "Hitting Adjustments": [],
        "Coaching Insights": []
    }
    
    # Sort clusters by pitcher success rate
    stance_analysis_sorted = stance_analysis.sort_values('pitcher_win_rate', ascending=False)
    
    most_vulnerable = stance_analysis_sorted.iloc[0]  # Highest pitcher win rate
    least_vulnerable = stance_analysis_sorted.iloc[-1]  # Lowest pitcher win rate
    
    # Pitching recommendations
    recommendations["Pitching Strategy"].extend([
        f"Target Cluster {int(most_vulnerable['stance_cluster'])} stances (foot sep: {most_vulnerable['avg_foot_separation']:.1f}, "
        f"angle: {most_vulnerable['avg_stance_angle']:.1f}°) - {most_vulnerable['pitcher_win_rate']:.1%} success rate",
        
        f"Avoid attacking Cluster {int(least_vulnerable['stance_cluster'])} stances directly - "
        f"only {least_vulnerable['pitcher_win_rate']:.1%} pitcher success rate",
    ])
    
    # Add pitch-specific recommendations if available
    for _, row in stance_analysis.iterrows():
        cluster = int(row['stance_cluster'])
        if 'best_pitch_vs_stance' in row and pd.notna(row['best_pitch_vs_stance']):
            best_pitch = row['best_pitch_vs_stance']
            success_rate = row['best_pitch_success_rate']
            recommendations["Pitching Strategy"].append(
                f"Use {best_pitch} vs Cluster {cluster} - {success_rate:.1%} effectiveness"
            )
    
    # Hitting recommendations
    recommendations["Hitting Adjustments"].extend([
        f"Players with narrow stances should consider widening (avg foot sep < 27)",
        f"Extremely closed stances (< -20°) may benefit from slight opening",
        f"Wide stances (> 35 foot separation) show good overall performance"
    ])
    
    # Add specific stance recommendations
    for _, row in stance_analysis.iterrows():
        cluster = int(row['stance_cluster'])
        if row['hitter_win_rate'] < 0.25:  # Low hitter success
            angle = row['avg_stance_angle']
            foot_sep = row['avg_foot_separation']
            recommendations["Hitting Adjustments"].append(
                f"Cluster {cluster} hitters: Consider adjusting from {angle:.1f}° angle, "
                f"{foot_sep:.1f} foot separation for better outcomes"
            )
    
    # Coaching insights
    recommendations["Coaching Insights"].extend([
        f"Stance clustering reveals {len(stance_analysis)} distinct batting approaches",
        f"Best performing stance: {most_vulnerable['avg_foot_separation']:.1f} foot separation, "
        f"{most_vulnerable['avg_stance_angle']:.1f}° angle",
        f"Success rate variance across stances: "
        f"{stance_analysis['pitcher_win_rate'].std():.3f} (higher = more differentiation)"
    ])
    
    return recommendations


def get_player_stance_recommendation(player_name: str, df: pd.DataFrame, 
                                   stance_analysis: pd.DataFrame) -> str:
    """Generate a player-centric summary and guidance from their dominant stance cluster.

    Args:
        player_name (str): Name formatted to match df['name'] (e.g., 'Olson, Matt').
        df (pd.DataFrame): Processed dataset containing 'name' and 'stance_cluster'.
        stance_analysis (pd.DataFrame): Cluster-level performance summary.

    Returns:
        str: Readable multi-line recommendation with stance characteristics and
            high-level advice based on cluster performance.
    """
    
    player_data = df[df['name'] == player_name]
    
    if player_data.empty:
        return f"No data found for player: {player_name}"
    
    if 'stance_cluster' not in player_data.columns:
        return "No stance cluster data available"
    
    player_cluster = player_data['stance_cluster'].mode()
    if player_cluster.empty:
        return "No stance cluster determined for this player"
    
    cluster = int(player_cluster.iloc[0])
    cluster_info = stance_analysis[stance_analysis['stance_cluster'] == cluster]
    
    if cluster_info.empty:
        return f"No analysis available for stance cluster {cluster}"
    
    cluster_row = cluster_info.iloc[0]
    
    recommendation = f"""
Player: {player_name}
Stance Cluster: {cluster}
Cluster Performance: {cluster_row['pitcher_win_rate']:.1%} pitcher success rate

Current Stance Characteristics:
- Foot Separation: {cluster_row['avg_foot_separation']:.1f}
- Stance Angle: {cluster_row['avg_stance_angle']:.1f}°

Recommendations:
"""
    
    if cluster_row['pitcher_win_rate'] > 0.4:  # High pitcher success = bad for hitter
        recommendation += "- Consider stance adjustments - current stance is vulnerable\n"
        recommendation += "- Work on timing vs dominant pitch types in this cluster\n"
    else:
        recommendation += "- Maintain current stance approach - shows good results\n"
        recommendation += "- Focus on consistency rather than major changes\n"
    
    if 'best_pitch_vs_stance' in cluster_row and pd.notna(cluster_row['best_pitch_vs_stance']):
        recommendation += f"- Be especially alert for {cluster_row['best_pitch_vs_stance']} pitches\n"
    
    return recommendation
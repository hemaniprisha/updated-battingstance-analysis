"""
Baseball Analytics System

A comprehensive machine learning system for predicting MLB pitch outcomes
and identifying strategic batting stance vulnerabilities using Statcast data
and biomechanical batting stance measurements.

Author: Prisha Hemani
Version: 2.0
"""

from .data_processing import ImprovedBaseballSystem
from .models import ImprovedPitchClassifier
from .analysis import perform_stance_clustering, analyze_stance_performance, generate_recommendations
from .visualization import plot_results, export_results

__version__ = "2.0"
__author__ = "Prisha Hemani"

__all__ = [
    'ImprovedBaseballSystem',
    'ImprovedPitchClassifier',
    'perform_stance_clustering',
    'analyze_stance_performance', 
    'generate_recommendations',
    'plot_results',
    'export_results'
]
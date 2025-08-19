"""
Volcano Explorer Package

A modular MDP-based volcanic terrain exploration system.
"""

from .environment import VolcanicGridWorld
from .solver import value_iteration, extract_policy, evaluate_policy
from .visualization import (
    plot_terrain_map, plot_value_function, plot_policy_map,
    create_separate_visualizations, create_final_path_summary
)
from .evaluation import PerformanceEvaluator, run_comprehensive_evaluation

__version__ = "2.0.0"
__author__ = "CSE440 Group 5"

__all__ = [
    'VolcanicGridWorld',
    'value_iteration', 
    'extract_policy',
    'evaluate_policy',
    'create_separate_visualizations',
    'create_final_path_summary',
    'plot_terrain_map',
    'plot_value_function', 
    'plot_policy_map',
    'PerformanceEvaluator',
    'run_comprehensive_evaluation'
]

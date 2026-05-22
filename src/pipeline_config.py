"""
Pipeline Configuration.

This module centralizes all hyperparameters and logic-related settings
for the cognitive architecture, Q-Learning, and state management.
Modify these values to experiment with the agent's behavior.
"""

from typing import Tuple

# --- Goal State ---
# The emotional equilibrium state (Arousal, Valence) the agent aims to reach.
GOAL_STATE_AROUSAL_VALENCE: Tuple[float, float] = (0.3, 0.3)

# --- Q-Learning Parameters ---
# Number of bins per axis. 
# (e.g., 10 means 10x10 grid; 21 means precise steps of 0.1 from -1.0 to 1.0) which also means slower learning
GRID_SIZE: int = 21

# Learning Rate (alpha): How much new information overrides old information (0 to 1).
LEARNING_RATE: float = 0.1

# Discount Factor (gamma): Importance of future rewards (0 to 1).
DISCOUNT_FACTOR: float = 0.9

# Exploration Rate (epsilon): Probability of choosing a random action (0 to 1).
EXPLORATION_RATE: float = 0.1

# --- Appraisal Parameters ---
# Tolerance threshold (psi): Allowed distance from goal state before regulation is triggered.
TOLERANCE_THRESHOLD: float = 0.25
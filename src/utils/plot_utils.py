"""
Utility functions for creating plots and visualizations.
"""

import matplotlib.pyplot as plt
from typing import List

def create_emotion_trajectory_plot(valence_history: List[float], arousal_history: List[float]):
    """
    Creates a 2D plot showing the trajectory of emotional states.

    Args:
        valence_history (List[float]): A list of valence scores.
        arousal_history (List[float]): A list of arousal scores.

    Returns:
        matplotlib.figure.Figure: The plot figure, or None if no data.
    """
    if not valence_history:
        return None

    # Use local variables instead of st.session_state
    valence = valence_history
    arousal = arousal_history

    # If we only have one data point, duplicate it to show a point
    if len(valence) == 1:
        valence = [valence[0], valence[0]]
        arousal = [arousal[0], arousal[0]]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the points
    ax.scatter(valence, arousal, color='royalblue', zorder=3)

    # Connect the points with a line to show trajectory
    if len(valence) > 1:
        ax.plot(valence, arousal, linestyle='--', color='gray', zorder=2)

    # Annotate each point with its index (step number)
    for i, (x, y) in enumerate(zip(valence, arousal)):
        ax.annotate(f'{i+1}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)

    # Add quadrant lines
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Set axis limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f7f7f7')

    plt.tight_layout()
    return fig
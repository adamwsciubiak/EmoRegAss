"""
Implementation of Pico's planner agent accordingly to the paper.

This file defines the data structures for the static planner derived from the paper (Section 4.2.).
Correlations of personality traits and emotion regulation techniques are refined based on 
Table 1 of Pico et al. (2024).

Glossary:
alpha (α)               - actions / actions catalog: one of the five strategies defined in Section 2.1
state (S)               - emotional state: a point in the 2D Arousal-Valence space
delta_Sa (ΔSα)          - change in State in the effect of the Action
personality_weights (θα)- "personality weights", correlation between the OCEAN traits and regulation strategies.
                          '+' = 1.0, '-' = -1.0, '0' = 0.0 (per Table 1)
trait (t)               - personality trait
epsilon_S (ε)           - emotional equilibrium state
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List

EmotionalState = Tuple[float, float]  # (Arousal, Valence)

@dataclass
class RegulationAction:
    """Represents a single emotion regulation action from the agent's catalog."""
    name: str
    strategy_family: str  # From Gross's model, e.g., 'Attentional Deployment'
    delta_Sa: EmotionalState  # Expected change in State (ΔSα)
    personality_weights: Dict[str, float]  # (θα) Correlation with OCEAN traits


# Data strictly derived from Table 1 and Table 2 in Pico et al. (2024).
ACTION_CATALOG: List[RegulationAction] = [
    RegulationAction(
        name="Avoidance",
        strategy_family="Situation Selection",
        delta_Sa=(-0.1, 0.2),
        # Table 1: O(-), C(+), E(-), A(0), N(+)
        personality_weights={"openness": -1.0, "conscientiousness": 1.0, "extraversion": -1.0, "agreeableness": 0.0, "neuroticism": 1.0}
    ),
    RegulationAction(
        name="Self-assertion",
        strategy_family="Situation Modification",
        delta_Sa=(0.1, 0.3),
        # Table 1: O(+), C(+), E(+), A(-), N(-)
        personality_weights={"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": -1.0, "neuroticism": -1.0}
    ),
    RegulationAction(
        name="Distraction",
        strategy_family="Attentional Deployment",
        delta_Sa=(-0.3, 0.2),
        # Table 1: O(+), C(+), E(0), A(0), N(-)
        personality_weights={"openness": 1.0, "conscientiousness": 1.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": -1.0}
    ),
    RegulationAction(
        name="Reappraisal",
        strategy_family="Cognitive Change",
        delta_Sa=(-0.1, 0.3),
        # Table 1: O(+), C(0), E(0), A(0), N(-)
        personality_weights={"openness": 1.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": -1.0}
    ),
    RegulationAction(
        name="Suppression",
        strategy_family="Response Modulation",
        delta_Sa=(-0.2, 0.1),
        # Table 1: O(0), C(0), E(-), A(0), N(0)
        personality_weights={"openness": 0.0, "conscientiousness": 0.0, "extraversion": -1.0, "agreeableness": 0.0, "neuroticism": 0.0}
    ),
]
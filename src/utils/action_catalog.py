""" 

Implementation of Pico's planner agent accordingly to the paper

This file defines the data structures for the static planner derived from the paper (Section 4.2.).
Correlations of personality traits and emotion regulation techniques are yet to be refined with more precise based on Barańczuk (2019) or Hughes et al., (2020)

Glossery:

alpha               actions / actions catalog       (α),            a one of the five strategies defined in Section 2.1" (Section 4, para. 2)
state               emotional state                 (S),            a point in the 2D Arousal-Valence space (Section 2.2, Figure 3) -> tuple
delta_Sa            delta S alpha                   (ΔSα),          change in State in the effect of the Action (Section 4, para. 6; Table 2)
theta_a             theta alpha                     (θα),           "personality weights", correlation between the personality traits of the OCEAN model 
                                                                    and the regulation strategies (Table 1). In the current version: '+' = 1.0, '-' = -1.0.


perTrait            personality trait               (t)
epsilon_S           emotional equilibrium state     (ε)


"""

from dataclasses import dataclass
from typing import Tuple, Dict, List


EmotionalState = Tuple[float, float]  # (Arousal, Valence)

@dataclass
class RegulationAction:
    """Represents a single emotion regulation action from the agent's catalog."""
    name: str
    strategy_family: str # From Gross's model, e.g., 'Attentional Deployment'

    delta_Sa: EmotionalState  # change in State in the effect of the Action
    theta_a: Dict[str, float] # "personality weights"


# This data is derived from Table 1 and Table 2 in Pico et al. (2024).
ACTION_CATALOG: List[RegulationAction] = [
    RegulationAction(
        name="Avoidance",
        strategy_family="Situation Selection",
        delta_Sa=(-0.1, 0.2), # Example from Table 2
        theta_a={"openness": -1.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 1.0}
    ),
    RegulationAction(
        name="Self-assertion",
        strategy_family="Situation Modification",
        delta_Sa=(0.1, 0.3), # Example from Table 2
        theta_a={"openness": 0.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": -1.0, "neuroticism": 0.0}
    ),
    RegulationAction(
        name="Distraction",
        strategy_family="Attentional Deployment",
        delta_Sa=(-0.3, 0.2), # Example from Table 2
        theta_a={"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": 0.0, "neuroticism": 0.0}
    ),
    RegulationAction(
        name="Reappraisal",
        strategy_family="Cognitive Change",
        delta_Sa=(-0.1, 0.3), # Example from Table 2
        theta_a={"openness": 1.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 1.0, "neuroticism": 0.0}
    ),

    RegulationAction(
        name="Suppression",
        strategy_family="Response Modulation",
        delta_Sa=(-0.2, 0.1), # Example from Table 2
        theta_a={"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 1.0}
    ),
]
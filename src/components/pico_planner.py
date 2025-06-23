""" 

Implementation of the static planner from Pico et al. (2024), Section 4.2.

This planner selects the best regulation action based on effectiveness and personality suitability, without a learning component.


Glossery:

alpha                   actions / actions catalog           (α)                                a one of the five strategies defined in Section 2.1" (Section 4, para. 2)
state                   emotional state                     (S)                                a point in the 2D Arousal-Valence space (Section 2.2, Figure 3) -> tuple
delta_Sa                delta S alpha                       (ΔSα)                              change in State in the effect of the Action (Section 4, para. 6; Table 2)
theta_a / 'weight'      theta alpha                         (θα)                               "personality weights", correlation between the personality traits of the OCEAN model 
                                                                                                        and the regulation strategies (Table 1). In the current version: '+' = 1.0, '-' = -1.0.
                                                                                
P_a                     P alpha                             Pa = (1/N) * Σ(θa * ti)             the suitability of the action α according to the user’s personality traits [-1,1]
                                                                                                        N is the number of personality traits (five in the case of the OCEAN model)
                                                                                                        Equation 4 in Pico.

a_hat                   alpha hat                           â = argmax(d(S'e,a, Sε)⁻¹ + Pa)     the best emo. reg. action (highest ΔSα) in a given emotional state (S) for given personality (Pa)
                                                                                                        d(S'e,a, Sε)⁻¹ is the "effectiveness" term. A smaller distance to the goal Sε results in a larger score
                                                                                                        This equation is implemented as select_action function below.
                                                                                                        Equation 2 and 3 in Pico, 
trait                   personality trait                   (t)
ti / "user_trait_score" personality trait of a given user   (ti)
epsilon_S               emotional equilibrium state         (ε)

d                       Euclidean distance                                                      standard "straight-line" distance in 2D space between two emotional states.
                                                                                                        calculated as math.sqrt((x2 - x1)**2 + (y2 - y1)**2) -> the equation is weird because it's based on the Pythagorean Theorem

"""

import math
from typing import Dict, Tuple

from src.utils.action_catalog import ACTION_CATALOG, RegulationAction, EmotionalState



# Define the user's personality as a type hint for clarity. OCEAN traits are keys, scores (0-1) are values.
PersonalityProfile = Dict[str, float]



def _calculate_distance(state1: EmotionalState, state2: EmotionalState) -> float:
    """Calculates the Euclidean distance between two emotional states."""
    arousal1, valence1 = state1
    arousal2, valence2 = state2
    return math.sqrt((arousal1 - arousal2)**2 + (valence1 - valence2)**2)



def _calculate_personality_score(action: RegulationAction, personality: PersonalityProfile) -> float: # The sum over all traits, and then divided by the total number  N
    """
    Implements Equation 4 from Pico et al. (2024) to calculate Pa.
    This measures the suitability of an action for a given personality.
    """
    N = len(action.personality_weights) # the number of personality traits (five in the case of the OCEAN model),

    total_score = 0
    for trait, weight in action.personality_weights.items(): #  iterating through the personality_weights dictionary of a single RegulationAction object. 
        # Get the user's score for that trait (ti) from the personality profile
        user_trait_score = personality.get(trait)
        
        # Calculate the product for this trait: θa * ti
        total_score += weight * user_trait_score
    
    # The paper implies Pa is in the range [-1, 1].
    # Since θa is in {-1, 0, 1} and ti is in [0, 1], the sum is in [-N, N].
    # HOWEVER dividing by N correctly scales the final Pa score to the [-1, 1] range.

    return total_score / N



class PicoPlanner:
    """
    A static planner that selects the best emotion regulation action based on
    pre-defined rules and personality, as per Pico et al. (2024), Section 4.2.
    """
    def __init__(self, action_catalog: list[RegulationAction] = ACTION_CATALOG):
        """Initializes the planner with a catalog of possible actions."""
        self.action_catalog = action_catalog

    def select_action(
        self,
        current_state: EmotionalState,
        goal_state: EmotionalState,
        personality: PersonalityProfile
    ) -> RegulationAction:
        """
        Selects the best action using Equation 2 from Pico et al. (2024).

        Args:
            current_state (Sa): The user's current (Arousal, Valence).
            goal_state (Sε): The user's target equilibrium (Arousal, Valence).
            personality (t): The user's OCEAN personality profile (scores 0-1).

        Returns:
            The single "best" RegulationAction to perform.
        """

        best_action = None
        max_score = -float('inf') # negative infinity


        for action in self.action_catalog:
            # --- Implement Equation 2: argmax(d(S'e,a, Sε)⁻¹ + Pa) ---

            # 1. Calculate the personality suitability score (Pa) using Equation 4
            pa_score = _calculate_personality_score(action, personality)

            # 2. Calculate the expected next state (S'e,a) using Equation 3
            # S'e,a = Sa + ΔSa
            expected_next_arousal = current_state[0] + action.delta_Sa[0]
            expected_next_valence = current_state[1] + action.delta_Sa[1]
            expected_next_state = (expected_next_arousal, expected_next_valence)

            # 3. Calculate the effectiveness score: d(S'e,a, Sε)⁻¹
            distance_to_goal = _calculate_distance(expected_next_state, goal_state)
            # Add a small epsilon (a very small value, not to be confused with emotional equalibrium marked as epsilon per Pico) to avoid division by zero if distance is 0
            effectiveness_score = 1 / (distance_to_goal + 1e-6)
            
            # 4. Calculate the final score
            total_score = effectiveness_score + pa_score    

            # 5. Check if this is the best action so far (argmax)
            if total_score > max_score:
                max_score = total_score
                best_action = action
        
        return best_action
    

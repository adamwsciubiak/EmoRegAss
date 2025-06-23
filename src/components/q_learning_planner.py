"""
Implementation of the dynamic, learning-based planner from Pico et al. (2024), Section 4.3.

This planner uses a Q-learning algorithm to personalize and improve its action selection
over time, learning from the outcomes of its interactions with the user. It builds upon
the static model by treating it as a "well-founded basis" for its initial knowledge.

Glossary (Q-Learning Specific):
Q-Table: A table mapping (state, action) pairs to a Q-value, which represents the
         expected cumulative future reward. Implemented as a NumPy array.
State (s): A discrete representation of the continuous Arousal-Valence space.
           Required for the Q-table.
Action (a): One of the regulation strategies from the action catalog.
Reward (r): A numerical feedback signal indicating the immediate outcome of an
            action.
Learning Rate (α): Controls how much new information overrides old information.
Discount Factor (γ): Determines the importance of future rewards.
Epsilon (ε): The probability of choosing a random action (exploration) vs. the
             best-known action (exploitation).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# --- Import from your existing project structure ---
# We use the static planner to perform the "warm start" initialization.
from .pico_planner import PicoPlanner, _calculate_personality_score 
from src.utils.action_catalog import ACTION_CATALOG, RegulationAction, EmotionalState

# Define the user's personality as a type hint for clarity
PersonalityProfile = Dict[str, float]

class QLearningPlanner:
    """
    An adaptive planner that uses Q-learning to dynamically improve emotion
    regulation strategies for a specific user, as described in
    Pico et al. (2024), Section 4.3.
    """
    def __init__(
        self,
        action_catalog: List[RegulationAction] = ACTION_CATALOG,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        grid_size: int = 10
    ):
        """
        Initializes the Q-learning planner.

        Args:
            action_catalog: A list of possible RegulationAction objects.
            learning_rate (α): The weight given to new information.
            discount_factor (γ): The importance of future rewards.
            exploration_rate (ε): The probability of choosing a random action.
            grid_size: The number of bins to discretize the Arousal/Valence space into.
                       A 10x10 grid results in 100 discrete states.
        """
        self.action_catalog = action_catalog
        self.action_map = {action.name: i for i, action in enumerate(self.action_catalog)}
        self.num_actions = len(self.action_catalog)

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

        self.grid_size = grid_size
        # The Q-table is a 2D array: (number of states) x (number of actions)
        # Each state is a cell in the (grid_size x grid_size) grid.
        self.q_table = np.zeros((grid_size * grid_size, self.num_actions))

        # Placeholder for the static planner used for initialization
        self._static_planner = PicoPlanner(self.action_catalog)

    def _discretize_state(self, state: EmotionalState) -> int:
        """
        Converts a continuous (Arousal, Valence) state into a discrete integer index.
        The Arousal/Valence space is [-1, 1]. We map this to grid indices [0, grid_size-1].

        Reference: Pico et al. (2024), Section 4.3. The paper frames the problem as an
        MDP, which necessitates discretizing the state space for a Q-table implementation.
        """
        arousal, valence = state
        # Normalize state from [-1, 1] to [0, 1]
        arousal_norm = (arousal + 1) / 2
        valence_norm = (valence + 1) / 2

        # Scale to grid size and convert to integer bins
        arousal_bin = int(arousal_norm * (self.grid_size - 1))
        valence_bin = int(valence_norm * (self.grid_size - 1))

        # Flatten the 2D grid coordinate to a single state index
        state_index = arousal_bin * self.grid_size + valence_bin
        return state_index

    def initialize_q_table(self, goal_state: EmotionalState, personality: PersonalityProfile):
        """
        Performs a "warm start" of the Q-table using the static PicoPlanner.

        Reference: Pico et al. (2024), Section 4.3. "Initially, these values in the
        Q-table reflect the general knowledge... the learning algorithm does not start
        from scratch, but from a well-founded basis."
        """
        print("INFO: Performing warm start initialization of the Q-table...")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Convert grid cell back to an approximate continuous state
                arousal = (row / (self.grid_size - 1)) * 2 - 1
                valence = (col / (self.grid_size - 1)) * 2 - 1
                current_state_approx = (arousal, valence)

                state_index = self._discretize_state(current_state_approx)

                # Use the static planner's logic to get initial scores for each action
                for i, action in enumerate(self.action_catalog):
                    # Identical logic to PicoPlanner.select_action's loop
                    pa_score = _calculate_personality_score(action, personality)
                    
                    expected_next_arousal = current_state_approx[0] + action.delta_Sa[0]
                    expected_next_valence = current_state_approx[1] + action.delta_Sa[1]
                    expected_next_state = (expected_next_arousal, expected_next_valence)
                    
                    distance_to_goal = np.sqrt((expected_next_state[0] - goal_state[0])**2 + (expected_next_state[1] - goal_state[1])**2)
                    effectiveness_score = 1 / (distance_to_goal + 1e-6)
                    
                    # The initial Q-value is the score from Equation 2
                    self.q_table[state_index, i] = effectiveness_score + pa_score
        print("INFO: Q-table initialization complete.")

    def select_action(self, current_state: EmotionalState) -> RegulationAction:
        """
        Selects an action using an epsilon-greedy policy based on the Q-table.

        Args:
            current_state (Sa): The user's current continuous (Arousal, Valence).

        Returns:
            The chosen RegulationAction object.
        """
        state_idx = self._discretize_state(current_state)

        # Epsilon-greedy: Explore or Exploit
        if np.random.uniform(0, 1) < self.epsilon:
            # --- Exploration: Choose a random action ---
            action_idx = np.random.choice(self.num_actions)
        else:
            # --- Exploitation: Choose the best-known action from the Q-table ---
            action_idx = np.argmax(self.q_table[state_idx, :])

        return self.action_catalog[action_idx]

    def update_q_table(
        self,
        prev_state: EmotionalState,
        action: RegulationAction,
        reward: float,
        next_state: EmotionalState
    ):
        """
        Updates the Q-table using the Bellman equation after an action is taken.

        Reference: Pico et al. (2024), Section 4.3. "The updating of this table follows
        a formula defined for Q-learning, which weights the immediate reward... with
        the estimate of future reward."

        Args:
            prev_state: The continuous state before the action was taken.
            action: The RegulationAction that was executed.
            reward: The immediate numerical reward received after the action.
            next_state: The new continuous state observed after the action.
        """
        # Get discrete indices for states and action
        prev_state_idx = self._discretize_state(prev_state)
        next_state_idx = self._discretize_state(next_state)
        action_idx = self.action_map[action.name]

        # --- Bellman Equation ---
        # Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]
        
        old_value = self.q_table[prev_state_idx, action_idx]
        next_max = np.max(self.q_table[next_state_idx, :]) # Best Q-value for the next state

        # The core update formula
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[prev_state_idx, action_idx] = new_value

    def get_q_table_as_dataframe(self) -> pd.DataFrame:
        """Utility function to view the Q-table for debugging."""
        action_names = [action.name for action in self.action_catalog]
        return pd.DataFrame(self.q_table, columns=action_names)
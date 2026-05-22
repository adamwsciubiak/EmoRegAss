"""
Dynamic Planner (Cognitive Core) for the Emotion Regulation Assistant.

This module unifies the static rules (Pico et al., 2024) and dynamic Q-Learning.
It strictly follows a 3-step cognitive process:
1. evaluate(): Critic cycle - calculates reward, updates Q-Table via Markov Decision Process (episodic_memory.py)
2. strategize(): Actor cycle - selects the optimal strategy using Q-Table.
3. tactics(): Retrieval cycle - fetches the concrete step-by-step plan via RAG.
"""

import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any

from src.static_knowledge.action_catalog import ACTION_CATALOG, RegulationAction, EmotionalState
from src.dynamic_knowledge.sensory_memory import SensoryMemory
from src.dynamic_knowledge.q_learning.q_table_manager import QTableManager

logger = logging.getLogger(__name__)
PersonalityProfile = Dict[str, float]

class DynamicPlanner:
    """
    The central cognitive module responsible for evaluating states and 
    planning emotion regulation interventions.
    """
    
    def __init__(
        self,
        q_manager: QTableManager,
        rag_retriever: Any,
        action_catalog: List[RegulationAction] = ACTION_CATALOG,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        grid_size: int = 21,
        tolerance_threshold: float = 0.25
    ):
        self.action_catalog = action_catalog
        self.action_map = {action.name: i for i, action in enumerate(self.action_catalog)}
        self.num_actions = len(self.action_catalog)
        
        self.q_manager = q_manager
        self.rag_retriever = rag_retriever
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.grid_size = grid_size
        self.tolerance_threshold = tolerance_threshold
        
        # --- ZABEZPIECZENIE WYMIARÓW (SHAPE VALIDATION) ---
        expected_shape = (grid_size * grid_size, self.num_actions)
        loaded_q = self.q_manager.load_q_table()
        
        if loaded_q is not None:
            # Sprawdzamy, czy wczytana tablica pasuje do aktualnych parametrów siatki
            if loaded_q.shape == expected_shape:
                self.q_table = loaded_q
                self.is_calibrated = True
                logger.info("Loaded Q-Table shape matches current grid_size.")
            else:
                logger.warning(f"Saved Q-Table shape {loaded_q.shape} does not match expected {expected_shape}. Discarding old Q-Table (grid_size was likely changed).")
                self.q_table = np.zeros(expected_shape)
                self.is_calibrated = False
        else:
            self.q_table = np.zeros(expected_shape)
            self.is_calibrated = False
            
    # --- HELPER MATHEMATICAL FUNCTIONS ---
    
    def _discretize_state(self, state: EmotionalState) -> int:
        arousal, valence = state
        arousal_norm = (arousal + 1) / 2
        valence_norm = (valence + 1) / 2
        arousal_bin = min(int(arousal_norm * self.grid_size), self.grid_size - 1)
        valence_bin = min(int(valence_norm * self.grid_size), self.grid_size - 1)
        return arousal_bin * self.grid_size + valence_bin

    def _calculate_distance(self, state1: EmotionalState, state2: EmotionalState) -> float:
        return math.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)

    def _calculate_personality_score(self, action: RegulationAction, personality: PersonalityProfile) -> float:
        N = len(action.personality_weights)
        total_score = sum(weight * personality.get(trait, 0.5) for trait, weight in action.personality_weights.items())
        return total_score / N

    # --- WARM START (INITIALIZATION) ---

    def calibrate(self, goal_state: EmotionalState, personality: PersonalityProfile) -> None:
        """
        Performs a 'warm start' of the Q-table based on mathematical heuristics.
        """
        logger.info("Calibrating Dynamic Planner (Warm Start)...")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                arousal = (row / (self.grid_size - 1)) * 2 - 1
                valence = (col / (self.grid_size - 1)) * 2 - 1
                current_state_approx = (arousal, valence)
                state_index = self._discretize_state(current_state_approx)

                for i, action in enumerate(self.action_catalog):
                    pa_score = self._calculate_personality_score(action, personality)
                    
                    expected_next_state = (
                        current_state_approx[0] + action.delta_Sa[0],
                        current_state_approx[1] + action.delta_Sa[1]
                    )
                    distance_to_goal = self._calculate_distance(expected_next_state, goal_state)
                    
                    # SCIENTIFIC FIX: Added + 1.0 to denominator to stabilize the scaling.
                    # Without this, small distances blow up to infinity, ignoring pa_score.
                    effectiveness_score = 1.0 / (distance_to_goal + 1.0)
                    
                    self.q_table[state_index, i] = effectiveness_score + pa_score
                    
        action_names = [a.name for a in self.action_catalog]
        self.q_manager.save_q_table(self.q_table, action_names)
        self.is_calibrated = True
        logger.info("Calibration complete. Q-Table saved.")

    # --- CORE COGNITIVE CYCLE ---

    def evaluate(self, current_state: EmotionalState, goal_state: EmotionalState, sensory_memory: SensoryMemory) -> bool:
        """
        STEP 1: EVALUATION (Critic)
        Calculates reward from the previous step, updates Q-Table, and decides if regulation is needed.
        
        Returns: True if user needs regulation (distance > threshold), False otherwise.
        """
        # 1. Update Learning (if we have a past state)
        prev_state, prev_action = sensory_memory.get_last_experience()
        
        if prev_state is not None and prev_action is not None:
            # Reward: How much closer did we get to the goal?
            dist_before = self._calculate_distance(prev_state, goal_state)
            dist_now = self._calculate_distance(current_state, goal_state)
            reward = dist_before - dist_now
            
            # Bellman Equation Update
            prev_state_idx = self._discretize_state(prev_state)
            curr_state_idx = self._discretize_state(current_state)
            action_idx = self.action_map[prev_action.name]
            
            old_q = self.q_table[prev_state_idx, action_idx]
            next_max_q = np.max(self.q_table[curr_state_idx, :])
            new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
            
            self.q_table[prev_state_idx, action_idx] = new_q
            logger.info(f"Evaluator updated Q-Table. Reward: {reward:.4f}")
            
            # Persist learning
            action_names = [a.name for a in self.action_catalog]
            self.q_manager.save_q_table(self.q_table, action_names)
            
        # 2. Check Appraisal Gap (Do we need to intervene?)
        distance_to_goal = self._calculate_distance(current_state, goal_state)
        needs_regulation = distance_to_goal > self.tolerance_threshold
        logger.info(f"Appraisal Gap: {distance_to_goal:.4f} (Threshold: {self.tolerance_threshold}). Needs regulation: {needs_regulation}")
        
        return needs_regulation

    def strategize(self, current_state: EmotionalState) -> RegulationAction:
        """
        STEP 2: STRATEGY (Actor)
        Selects an abstract emotion regulation strategy using epsilon-greedy policy.
        """
        state_idx = self._discretize_state(current_state)

        if np.random.uniform(0, 1) < self.epsilon:
            # Explore
            action_idx = np.random.choice(self.num_actions)
            logger.debug("Strategy: Exploration (Random)")
        else:
            # Exploit
            action_idx = np.argmax(self.q_table[state_idx, :])
            logger.debug("Strategy: Exploitation (Q-Table Max)")

        chosen_action = self.action_catalog[action_idx]
        logger.info(f"Strategist selected strategy: {chosen_action.name}")
        return chosen_action

    def tactics(self, chosen_strategy: RegulationAction) -> Dict[str, Any]:
        """
        STEP 3: TACTICS
        Converts the abstract strategy into a concrete step-by-step plan by querying DB2 (RAG).
        Notice: User message is completely absent here!
        """
        logger.info(f"Tactician requesting RAG manual for: {chosen_strategy.name}")
        
        # Fetch knowledge from Static Knowledge Base
        retrieved_steps = self.rag_retriever.retrieve_technique_manual(chosen_strategy.name)
        
        # Formulate a dry, strict plan for the Executor
        response_plan = {
            "goal": f"Introduce and guide the user through the '{chosen_strategy.name}' technique.",
            "technique_family": chosen_strategy.strategy_family,
            "steps": retrieved_steps
        }
        return response_plan
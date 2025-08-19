"""
Volcanic Grid World Environment

This module defines the world environment for the MDP-based volcanic explorer.
It manages the state of the world, including the grid map, terrain types, rewards, and dimensions.
"""

import numpy as np


class VolcanicGridWorld:
    """
    Volcanic Grid World Environment for MDP-based autonomous exploration
    
    Grid map terrain types:
    0: Unexplored, 1: Safe, 2: Gas Vent, 3: Lava (Terminal), 4: Crater (Terminal), 5: Goal (Terminal)
    """
    
    def __init__(self, grid_map, rewards, transition_prob, goal_pos=None):
        """
        Initialize the Volcanic Grid World environment
        
        Args:
            grid_map: 2D numpy array representing terrain types
            rewards: Dictionary mapping terrain types to reward values
            transition_prob: Dictionary with transition probabilities
            goal_pos: Tuple (row, col) for goal position, defaults to bottom-right
        """
        self.grid_map = grid_map.copy()
        self.original_map = grid_map.copy()  # Keep original for visualization
        self.height, self.width = grid_map.shape
        self.rewards = rewards
        self.transition_prob = transition_prob
        
        # Set goal position (default to bottom-right corner)
        if goal_pos is None:
            self.goal_pos = (self.height - 1, self.width - 1)
        else:
            self.goal_pos = goal_pos
        
        # Mark goal position in the grid (terrain type 5 = Goal)
        # Only if it's not already a hazard
        if self.grid_map[self.goal_pos[0], self.goal_pos[1]] not in [3, 4]:  # Not lava or crater
            self.grid_map[self.goal_pos[0], self.goal_pos[1]] = 5
            self.original_map[self.goal_pos[0], self.goal_pos[1]] = 5
        
        # Action space: {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.actions = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.num_actions = len(self.actions)
        
        # Define action deltas
        self.action_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Define slip directions (left and right relative to intended direction)
        self.slip_directions = {
            0: [2, 3],  # Up: slip Left or Right
            1: [3, 2],  # Down: slip Right or Left
            2: [0, 1],  # Left: slip Up or Down
            3: [1, 0]   # Right: slip Down or Up
        }
        
    def is_valid_state(self, row, col):
        """Check if state is within grid bounds"""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_terminal_state(self, row, col):
        """Check if state is terminal (Lava, Crater, or Goal)"""
        if not self.is_valid_state(row, col):
            return False
        terrain = self.grid_map[row, col]
        return terrain in [3, 4, 5]  # Lava, Crater, or Goal
    
    def get_next_state(self, state, action):
        """Get next state given current state and action"""
        row, col = state
        delta_row, delta_col = self.action_deltas[action]
        next_row, next_col = row + delta_row, col + delta_col
        
        # If next state is out of bounds, stay in current state
        if not self.is_valid_state(next_row, next_col):
            return (row, col)
        
        return (next_row, next_col)
    
    def get_reward(self, state, action, next_state):
        """
        Get reward for transitioning from state to next_state via action
        
        Rewards:
        - Moving into Unexplored cell: +20 (becomes Safe after visiting)
        - Moving into Gas Vent: -50
        - Moving into Lava or Crater: -1000
        - Moving into Goal: +200 (mission success)
        - Cost of living (other moves): -1
        - Small distance bonus towards goal: +0.1 per cell closer
        """
        next_row, next_col = next_state
        terrain = self.grid_map[next_row, next_col]
        
        # Base reward based on terrain type
        if terrain == 0:  # Unexplored
            # Update terrain to Safe after visiting
            self.grid_map[next_row, next_col] = 1
            base_reward = self.rewards['unexplored']
        elif terrain == 1:  # Safe
            base_reward = self.rewards['safe']
        elif terrain == 2:  # Gas Vent
            base_reward = self.rewards['gas_vent']
        elif terrain == 3:  # Lava (Terminal)
            base_reward = self.rewards['lava']
        elif terrain == 4:  # Crater (Terminal)
            base_reward = self.rewards['crater']
        elif terrain == 5:  # Goal (Terminal)
            base_reward = self.rewards.get('goal', 200)  # Large positive reward for reaching goal
        else:
            base_reward = self.rewards['safe']  # Default
        
        # Add small distance incentive towards goal (helps guide exploration)
        if terrain not in [3, 4, 5]:  # Not terminal states
            current_distance = abs(state[0] - self.goal_pos[0]) + abs(state[1] - self.goal_pos[1])
            next_distance = abs(next_state[0] - self.goal_pos[0]) + abs(next_state[1] - self.goal_pos[1])
            distance_bonus = 0.1 * (current_distance - next_distance)  # Reward for getting closer
            base_reward += distance_bonus
        
        return base_reward
    
    def get_transition_probabilities(self, state, action):
        """
        Get transition probabilities for all possible next states
        
        Returns list of (probability, next_state, reward) tuples
        """
        transitions = []
        
        # Intended direction (probability 0.8)
        intended_next = self.get_next_state(state, action)
        reward = self.get_reward(state, action, intended_next)
        transitions.append((self.transition_prob['intended'], intended_next, reward))
        
        # Slip directions (probability 0.1 each)
        for slip_action in self.slip_directions[action]:
            slip_next = self.get_next_state(state, slip_action)
            slip_reward = self.get_reward(state, slip_action, slip_next)
            transitions.append((self.transition_prob['slip'], slip_next, slip_reward))
        
        return transitions
    
    def reset_to_original(self):
        """Reset grid map to original state (useful for multiple simulations)"""
        self.grid_map = self.original_map.copy()

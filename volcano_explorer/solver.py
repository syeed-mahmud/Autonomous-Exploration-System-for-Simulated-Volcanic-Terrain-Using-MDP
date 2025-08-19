"""
MDP Solver Module

This module contains the core algorithms for solving the Markov Decision Process.
It includes Value Iteration and Policy Extraction algorithms.
"""

import numpy as np


def value_iteration(grid_world, gamma=0.99, theta=1e-6, verbose=True):
    """
    Solve the MDP using Value Iteration algorithm
    
    Args:
        grid_world: VolcanicGridWorld instance
        gamma: Discount factor
        theta: Convergence threshold
        verbose: Whether to print iteration progress
        
    Returns:
        Converged value function as 2D numpy array
    """
    # Initialize value function
    V = np.zeros((grid_world.height, grid_world.width))
    
    iteration = 0
    if verbose:
        print("Starting Value Iteration...")
    
    while True:
        delta = 0
        V_old = V.copy()
        
        # Update value for each state
        for row in range(grid_world.height):
            for col in range(grid_world.width):
                state = (row, col)
                
                # Skip terminal states
                if grid_world.is_terminal_state(row, col):
                    continue
                
                # Calculate value for each action
                action_values = []
                for action in range(grid_world.num_actions):
                    # Get transition probabilities and rewards
                    transitions = grid_world.get_transition_probabilities(state, action)
                    
                    # Calculate expected value for this action
                    expected_value = 0
                    for prob, next_state, reward in transitions:
                        next_row, next_col = next_state
                        expected_value += prob * (reward + gamma * V_old[next_row, next_col])
                    
                    action_values.append(expected_value)
                
                # Update value with maximum expected value
                if action_values:  # Only if there are valid actions
                    V[row, col] = max(action_values)
                    delta = max(delta, abs(V[row, col] - V_old[row, col]))
        
        iteration += 1
        if verbose and iteration % 100 == 0:
            print(f"Value Iteration - Iteration {iteration}, Delta: {delta:.6f}")
        
        # Check convergence
        if delta < theta:
            if verbose:
                print(f"Value Iteration converged after {iteration} iterations")
            break
    
    return V


def extract_policy(grid_world, value_function, gamma=0.99):
    """
    Extract optimal policy from converged value function
    
    Args:
        grid_world: VolcanicGridWorld instance
        value_function: Converged value function
        gamma: Discount factor
        
    Returns:
        Optimal policy as 2D numpy array
    """
    policy = np.zeros((grid_world.height, grid_world.width), dtype=int)
    
    for row in range(grid_world.height):
        for col in range(grid_world.width):
            state = (row, col)
            
            # Skip terminal states
            if grid_world.is_terminal_state(row, col):
                policy[row, col] = -1  # No action for terminal states
                continue
            
            # Calculate expected value for each action
            action_values = []
            for action in range(grid_world.num_actions):
                # Get transition probabilities and rewards
                transitions = grid_world.get_transition_probabilities(state, action)
                
                # Calculate expected value for this action
                expected_value = 0
                for prob, next_state, reward in transitions:
                    next_row, next_col = next_state
                    expected_value += prob * (reward + gamma * value_function[next_row, next_col])
                
                action_values.append(expected_value)
            
            # Choose action with maximum expected value
            if action_values:
                policy[row, col] = np.argmax(action_values)
    
    return policy


def evaluate_policy(grid_world, policy, gamma=0.99, theta=1e-6):
    """
    Evaluate a given policy to compute its value function
    
    Args:
        grid_world: VolcanicGridWorld instance
        policy: Policy to evaluate (2D numpy array)
        gamma: Discount factor
        theta: Convergence threshold
        
    Returns:
        Value function for the given policy
    """
    V = np.zeros((grid_world.height, grid_world.width))
    
    while True:
        delta = 0
        V_old = V.copy()
        
        for row in range(grid_world.height):
            for col in range(grid_world.width):
                state = (row, col)
                
                # Skip terminal states
                if grid_world.is_terminal_state(row, col):
                    continue
                
                # Get action from policy
                action = policy[row, col]
                if action == -1:  # No action for terminal states
                    continue
                
                # Get transition probabilities and rewards
                transitions = grid_world.get_transition_probabilities(state, action)
                
                # Calculate expected value for this action
                expected_value = 0
                for prob, next_state, reward in transitions:
                    next_row, next_col = next_state
                    expected_value += prob * (reward + gamma * V_old[next_row, next_col])
                
                V[row, col] = expected_value
                delta = max(delta, abs(V[row, col] - V_old[row, col]))
        
        # Check convergence
        if delta < theta:
            break
    
    return V

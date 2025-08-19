"""
Main Entry Point for Volcanic Terrain Explorer

This is the main orchestration script that brings together all components
of the MDP-based volcanic terrain exploration system.
"""

import numpy as np
from environment import VolcanicGridWorld
from solver import value_iteration, extract_policy
from visualization import create_separate_visualizations, create_final_path_summary
from evaluation import run_comprehensive_evaluation


def simulate_agent(grid_world, policy, start_pos, max_steps=100):
    """
    Simulate agent following the optimal policy
    
    Args:
        grid_world: VolcanicGridWorld instance
        policy: Optimal policy
        start_pos: Starting position (row, col)
        max_steps: Maximum number of steps
        
    Returns:
        path: List of states visited
        total_reward: Total accumulated reward
    """
    # Reset grid to original state for clean simulation
    grid_world.reset_to_original()
    
    path = [start_pos]
    total_reward = 0
    current_state = start_pos
    
    print(f"Starting simulation from position {start_pos}")
    
    for step in range(max_steps):
        row, col = current_state
        
        # Check if terminal state
        if grid_world.is_terminal_state(row, col):
            print(f"Reached terminal state at {current_state} after {step} steps")
            break
        
        # Get action from policy
        action = policy[row, col]
        if action == -1:  # No valid action
            print(f"No valid action at {current_state}")
            break
        
        # Execute action (deterministic for simulation)
        next_state = grid_world.get_next_state(current_state, action)
        reward = grid_world.get_reward(current_state, action, next_state)
        
        total_reward += reward
        current_state = next_state
        path.append(current_state)
        
        # Print progress for significant events
        if reward > 0:
            print(f"Step {step + 1}: Explored new cell at {current_state}, reward: +{reward}")
        elif reward < -10:
            print(f"Step {step + 1}: Encountered hazard at {current_state}, reward: {reward}")
        
        # Stop if reached same position (stuck)
        if len(path) > 1 and path[-1] == path[-2]:
            print(f"Agent stuck at {current_state}")
            break
    
    print(f"Simulation completed: {len(path)} steps, total reward: {total_reward:.1f}")
    return path, total_reward


def create_sample_environment():
    """
    Create a sample volcanic environment configuration
    
    Returns:
        grid_map: 2D numpy array representing the terrain
        rewards: Dictionary with reward values
        transition_prob: Dictionary with transition probabilities
    """
    # Define sample 10x10 grid map
    # 0: Unexplored, 1: Safe, 2: Gas Vent, 3: Lava (Terminal), 4: Crater (Terminal)
    grid_map = np.array([
        [0, 0, 1, 0, 0, 2, 0, 0, 1, 0],
        [0, 1, 0, 0, 2, 0, 0, 1, 0, 0],
        [1, 0, 0, 3, 0, 0, 1, 0, 0, 2],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 2, 0, 0, 4, 0, 0, 2, 0, 0],
        [1, 0, 0, 0, 0, 0, 3, 0, 0, 1],
        [0, 0, 1, 2, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 2],
        [2, 0, 0, 0, 3, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 2, 0, 0, 1]
    ])
    
    # Define rewards dictionary
    rewards = {
        'unexplored': 20,    # Moving into unexplored cell
        'safe': -1,          # Cost of living / moving into safe cell
        'gas_vent': -50,     # Moving into gas vent
        'lava': -1000,       # Moving into lava (terminal)
        'crater': -1000,     # Moving into crater (terminal)
        'goal': 200          # Large reward for reaching the goal (terminal)
    }
    
    # Define transition probabilities
    transition_prob = {
        'intended': 0.8,     # Probability of moving in intended direction
        'slip': 0.1          # Probability of slipping left or right
    }
    
    return grid_map, rewards, transition_prob


def print_environment_summary(grid_world):
    """Print a summary of the environment configuration"""
    print("🌋 Volcanic Terrain Autonomous Exploration using MDP")
    print("=" * 60)
    print("Grid World Configuration:")
    print(f"- Grid Size: {grid_world.height}x{grid_world.width}")
    print(f"- Unexplored cells: {np.sum(grid_world.original_map == 0)}")
    print(f"- Safe cells: {np.sum(grid_world.original_map == 1)}")
    print(f"- Gas vents: {np.sum(grid_world.original_map == 2)}")
    print(f"- Lava cells: {np.sum(grid_world.original_map == 3)}")
    print(f"- Crater cells: {np.sum(grid_world.original_map == 4)}")
    print(f"- Goal cells: {np.sum(grid_world.original_map == 5)}")
    print()


def main():
    """Main execution function"""
    # Create sample environment configuration
    grid_map, rewards, transition_prob = create_sample_environment()
    
    # Define start and goal positions for proper exploration
    start_pos = (0, 0)  # Top-left corner
    goal_pos = (grid_map.shape[0] - 1, grid_map.shape[1] - 1)  # Bottom-right corner
    
    # Create VolcanicGridWorld instance with explicit goal
    print("Initializing Volcanic Grid World...")
    grid_world = VolcanicGridWorld(grid_map, rewards, transition_prob, goal_pos=goal_pos)
    
    # Print environment summary
    print_environment_summary(grid_world)
    print(f"🎯 Goal Position: {goal_pos}")
    print(f"🚀 Start Position: {start_pos}")
    print(f"📏 Distance to Goal: {abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])} cells (Manhattan)")
    print("=" * 60)
    
    # Solve the MDP using Value Iteration
    print("Running Value Iteration...")
    value_function = value_iteration(grid_world, gamma=0.99, theta=1e-6, verbose=True)
    
    # Extract optimal policy
    print("\nExtracting Optimal Policy...")
    policy = extract_policy(grid_world, value_function, gamma=0.99)
    
    # Simulate agent following the policy
    print(f"\nSimulating agent path from {start_pos} to {goal_pos}...")
    path, total_reward = simulate_agent(grid_world, policy, start_pos)
    
    # Run comprehensive evaluation
    print("\n🔬 Running Comprehensive Model Evaluation...")
    evaluator = run_comprehensive_evaluation(grid_world, policy, value_function)
    
    # Generate Enhanced Visualizations
    print("\n📊 Generating Enhanced Visualizations...")
    create_separate_visualizations(grid_world, value_function, policy, path, total_reward, evaluator)
    
    # Generate Final Path Summary
    print("\n🎯 Generating Final Path Summary...")
    create_final_path_summary(grid_world, value_function, policy, path, total_reward, evaluator, start_pos)
    
    # Print completion summary
    print("\n✅ MDP Solution Complete!")
    print("🎯 FINAL RESULTS SUMMARY:")
    print("=" * 50)
    print("📈 Analysis Components:")
    print("  • Environment overview with terrain analysis")
    print("  • Value function analysis with statistics")
    print("  • Policy analysis with consistency metrics")
    print("  • Simulation analysis with efficiency metrics")
    print("  • Performance evaluation dashboard")
    print("  • Comparative analysis and correlations")
    print("  • 🎯 Final path summary with comprehensive results")
    
    # Print key metrics
    if evaluator and evaluator.evaluation_results:
        results = evaluator.evaluation_results
        if 'policy_quality' in results:
            qual = results['policy_quality']
            print(f"\n🎯 KEY PERFORMANCE METRICS:")
            print(f"  • Average Reward: {qual['avg_reward']:.2f}")
            print(f"  • Success Rate: {qual['success_rate']:.1%}")
            print(f"  • Exploration Rate: {qual['avg_exploration_rate']:.1%}")
            print(f"  • Safety Score: {qual['avg_safety_score']:.1%}")
        
        if 'convergence' in results:
            conv = results['convergence']
            print(f"  • Convergence Time: {conv['total_time']:.2f}s")
            print(f"  • Total Iterations: {conv['total_iterations']}")
    
    print("=" * 50)
    
    return {
        'grid_world': grid_world,
        'value_function': value_function,
        'policy': policy,
        'path': path,
        'total_reward': total_reward,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    results = main()

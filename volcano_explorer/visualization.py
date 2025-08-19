"""
Visualization Module

This module is responsible for all visual output of the MDP volcanic explorer.
It contains functions to create comprehensive visualizations of the environment,
value function, policy, and agent simulation with support for separate pages.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time


def plot_terrain_map(ax, grid_world, title="Volcanic Terrain"):
    """
    Plot the initial terrain map with color-coded terrain types
    
    Args:
        ax: Matplotlib axis object
        grid_world: VolcanicGridWorld instance
        title: Title for the subplot
    """
    # Define colors for terrain types
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    terrain_labels = ['Unexplored', 'Safe', 'Gas Vent', 'Lava', 'Crater', 'Goal']
    
    im = ax.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(6))
    cbar.set_ticklabels(terrain_labels)
    
    return im


def plot_value_function(ax, value_function, title="Converged Value Function"):
    """
    Plot the value function as a heatmap with numerical values
    
    Args:
        ax: Matplotlib axis object
        value_function: 2D numpy array of values
        title: Title for the subplot
    """
    im = ax.imshow(value_function, cmap='viridis')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add value text on each cell
    height, width = value_function.shape
    for i in range(height):
        for j in range(width):
            text = ax.text(j, i, f'{value_function[i, j]:.1f}',
                          ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    return im


def plot_policy_map(ax, grid_world, policy, title="Optimal Policy"):
    """
    Plot the optimal policy with arrows indicating directions
    
    Args:
        ax: Matplotlib axis object
        grid_world: VolcanicGridWorld instance
        policy: 2D numpy array representing the policy
        title: Title for the subplot
    """
    # Define colors for terrain types (with transparency)
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    
    im = ax.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add policy arrows
    arrow_symbols = ['↑', '↓', '←', '→']
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                action = policy[i, j]
                ax.text(j, i, arrow_symbols[action], ha="center", va="center", 
                       fontsize=12, fontweight='bold', color='black')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    return im


def plot_simulated_path(ax, grid_world, path, total_reward, title_prefix="Simulated Path"):
    """
    Plot the agent's simulated path on the terrain
    
    Args:
        ax: Matplotlib axis object
        grid_world: VolcanicGridWorld instance
        path: List of (row, col) tuples representing the path
        total_reward: Total accumulated reward
        title_prefix: Prefix for the subplot title
    """
    # Define colors for terrain types (with transparency)
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    
    im = ax.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.7)
    ax.set_title(f'{title_prefix} (Total Reward: {total_reward:.1f})', fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Draw path
    if len(path) > 1:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.8, label='Path')
        
        # Mark start and end
        start_pos = path[0]
        end_pos = path[-1]
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
        ax.plot(end_pos[1], end_pos[0], 'ro', markersize=10, label='End')
        
        ax.legend()
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    return im





def create_animation_frames(grid_world, policy, start_pos, max_steps=100):
    """
    Create frames for animating the agent's exploration
    
    Args:
        grid_world: VolcanicGridWorld instance
        policy: Optimal policy
        start_pos: Starting position
        max_steps: Maximum number of steps
        
    Returns:
        List of frames, each containing the grid state and agent position
    """
    frames = []
    current_state = start_pos
    current_grid = grid_world.original_map.copy()
    
    for step in range(max_steps):
        # Save current frame
        frames.append({
            'step': step,
            'grid': current_grid.copy(),
            'agent_pos': current_state,
            'path_so_far': frames.copy() if frames else []
        })
        
        row, col = current_state
        
        # Check if terminal state
        if grid_world.is_terminal_state(row, col):
            break
        
        # Get action from policy
        action = policy[row, col]
        if action == -1:  # No valid action
            break
        
        # Execute action (deterministic for visualization)
        next_state = grid_world.get_next_state(current_state, action)
        
        # Update grid if exploring new cell
        next_row, next_col = next_state
        if current_grid[next_row, next_col] == 0:  # Unexplored
            current_grid[next_row, next_col] = 1  # Mark as explored
        
        current_state = next_state
        
        # Stop if reached same position (stuck)
        if len(frames) > 1 and current_state == frames[-2].get('agent_pos'):
            break
    
    return frames


def plot_convergence_curve(iterations, deltas, title="Value Iteration Convergence"):
    """
    Plot the convergence curve of the value iteration algorithm
    
    Args:
        iterations: List of iteration numbers
        deltas: List of delta values for each iteration
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, deltas, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Max Value Change)')
    plt.title(title, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_separate_visualizations(grid_world, value_function, policy, path, total_reward, evaluation_results=None):
    """
    Create enhanced visualizations in separate windows with detailed analysis
    
    Args:
        grid_world: VolcanicGridWorld instance
        value_function: Converged value function
        policy: Optimal policy
        path: Simulated path
        total_reward: Total reward from simulation
        evaluation_results: Optional evaluation results for enhanced plots
    """
    # Set matplotlib to interactive mode for separate windows
    plt.ion()
    
    # 1. Environment Overview (Enhanced)
    create_environment_overview(grid_world)
    
    # 2. Value Function Analysis (Enhanced)
    create_value_function_analysis(grid_world, value_function)
    
    # 3. Policy Visualization (Enhanced)
    create_policy_analysis(grid_world, policy, value_function)
    
    # 4. Simulation Results (Enhanced)
    create_simulation_analysis(grid_world, path, total_reward)
    
    # 5. Performance Evaluation (New)
    if evaluation_results:
        create_evaluation_dashboard(evaluation_results)
    
    # 6. Comparative Analysis (New)
    create_comparative_analysis(grid_world, value_function, policy)
    
    plt.show(block=True)


def create_environment_overview(grid_world):
    """Create detailed environment overview in separate window"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🌋 Environment Overview & Analysis', fontsize=16, fontweight='bold')
    
    # Terrain colors and labels
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    terrain_labels = ['Unexplored', 'Safe', 'Gas Vent', 'Lava', 'Crater', 'Goal']
    
    # 1. Original Terrain Map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5)
    ax1.set_title('Original Terrain Map', fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=range(6))
    cbar1.set_ticklabels(terrain_labels)
    
    # 2. Terrain Statistics
    ax2 = axes[0, 1]
    terrain_counts = [np.sum(grid_world.original_map == i) for i in range(6)]
    terrain_percentages = [count/np.sum(terrain_counts)*100 for count in terrain_counts]
    
    bars = ax2.bar(terrain_labels, terrain_counts, color=terrain_colors)
    ax2.set_title('Terrain Distribution', fontweight='bold')
    ax2.set_ylabel('Number of Cells')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, pct in zip(bars, terrain_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    # 3. Hazard Proximity Map
    ax3 = axes[1, 0]
    hazard_proximity = calculate_hazard_proximity(grid_world)
    im3 = ax3.imshow(hazard_proximity, cmap='RdYlBu_r')
    ax3.set_title('Hazard Proximity Map', fontweight='bold')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, label='Distance to Nearest Hazard')
    
    # 4. Exploration Potential Map
    ax4 = axes[1, 1]
    exploration_potential = calculate_exploration_potential(grid_world)
    im4 = ax4.imshow(exploration_potential, cmap='viridis')
    ax4.set_title('Exploration Potential Map', fontweight='bold')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im4, ax=ax4, label='Nearby Unexplored Cells')
    
    plt.tight_layout()
    return fig


def create_value_function_analysis(grid_world, value_function):
    """Create detailed value function analysis in separate window"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📈 Value Function Analysis', fontsize=16, fontweight='bold')
    
    # 1. Value Function Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(value_function, cmap='viridis')
    ax1.set_title('Value Function Heatmap', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Value')
    
    # Add value contours
    ax1.contour(value_function, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    
    # 2. Value Function with Numbers
    ax2 = axes[0, 1]
    im2 = ax2.imshow(value_function, cmap='viridis')
    ax2.set_title('Value Function (Numerical)', fontweight='bold')
    
    # Add numerical values
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            text = ax2.text(j, i, f'{value_function[i, j]:.1f}',
                           ha="center", va="center", color="white", 
                           fontweight='bold', fontsize=8)
    
    # 3. Value Distribution Histogram
    ax3 = axes[0, 2]
    non_terminal_values = []
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                non_terminal_values.append(value_function[i, j])
    
    ax3.hist(non_terminal_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Value Distribution', fontweight='bold')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(non_terminal_values), color='red', linestyle='--', label=f'Mean: {np.mean(non_terminal_values):.2f}')
    ax3.legend()
    
    # 4. Value Gradient Magnitude
    ax4 = axes[1, 0]
    grad_y, grad_x = np.gradient(value_function)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    im4 = ax4.imshow(grad_magnitude, cmap='hot')
    ax4.set_title('Value Gradient Magnitude', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Gradient Magnitude')
    
    # 5. High-Value Regions
    ax5 = axes[1, 1]
    high_value_threshold = np.percentile(non_terminal_values, 75)
    high_value_mask = value_function > high_value_threshold
    
    # Show original terrain with high-value overlay
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    ax5.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.6)
    ax5.imshow(high_value_mask, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    ax5.set_title(f'High-Value Regions (>{high_value_threshold:.1f})', fontweight='bold')
    
    # 6. Value vs Distance to Hazards
    ax6 = axes[1, 2]
    hazard_distances = []
    values_by_distance = []
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                dist = calculate_min_hazard_distance(grid_world, i, j)
                hazard_distances.append(dist)
                values_by_distance.append(value_function[i, j])
    
    ax6.scatter(hazard_distances, values_by_distance, alpha=0.6, c='blue', s=20)
    ax6.set_xlabel('Distance to Nearest Hazard')
    ax6.set_ylabel('Value')
    ax6.set_title('Value vs Hazard Distance', fontweight='bold')
    
    # Add trend line
    if len(hazard_distances) > 1:
        z = np.polyfit(hazard_distances, values_by_distance, 1)
        p = np.poly1d(z)
        ax6.plot(sorted(hazard_distances), p(sorted(hazard_distances)), "r--", alpha=0.8)
    
    plt.tight_layout()
    return fig


def create_policy_analysis(grid_world, policy, value_function):
    """Create detailed policy analysis in separate window"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🎯 Policy Analysis', fontsize=16, fontweight='bold')
    
    # Terrain colors
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    
    # 1. Policy Arrows on Terrain
    ax1 = axes[0, 0]
    ax1.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.7)
    
    arrow_symbols = ['↑', '↓', '←', '→']
    arrow_colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                action = policy[i, j]
                ax1.text(j, i, arrow_symbols[action], ha="center", va="center", 
                        fontsize=12, fontweight='bold', color='black')
    
    ax1.set_title('Policy on Terrain', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Policy Action Distribution
    ax2 = axes[0, 1]
    action_counts = [0, 0, 0, 0]
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                action_counts[policy[i, j]] += 1
    
    bars = ax2.bar(action_names, action_counts, color=arrow_colors)
    ax2.set_title('Action Distribution', fontweight='bold')
    ax2.set_ylabel('Number of States')
    
    # Add count labels
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(count), ha='center', va='bottom')
    
    # 3. Policy Consistency Map
    ax3 = axes[0, 2]
    consistency_map = calculate_policy_consistency(grid_world, policy)
    im3 = ax3.imshow(consistency_map, cmap='RdYlGn')
    ax3.set_title('Policy Consistency', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Consistency Score')
    
    # 4. Value-Guided Policy
    ax4 = axes[1, 0]
    im4 = ax4.imshow(value_function, cmap='viridis', alpha=0.8)
    
    # Overlay arrows sized by value magnitude
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                action = policy[i, j]
                value_norm = (value_function[i, j] - np.min(value_function)) / (np.max(value_function) - np.min(value_function))
                size = 8 + 8 * value_norm  # Scale arrow size by value
                ax4.text(j, i, arrow_symbols[action], ha="center", va="center", 
                        fontsize=size, fontweight='bold', color='white')
    
    ax4.set_title('Value-Weighted Policy', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Value')
    
    # 5. Policy Entropy Map
    ax5 = axes[1, 1]
    entropy_map = calculate_policy_entropy(grid_world, policy, value_function)
    im5 = ax5.imshow(entropy_map, cmap='plasma')
    ax5.set_title('Policy Entropy', fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Entropy')
    
    # 6. Optimal vs Suboptimal Actions
    ax6 = axes[1, 2]
    optimality_scores = []
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                score = calculate_action_optimality(grid_world, policy, value_function, i, j)
                optimality_scores.append(score)
    
    ax6.hist(optimality_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax6.set_title('Action Optimality Distribution', fontweight='bold')
    ax6.set_xlabel('Optimality Score')
    ax6.set_ylabel('Frequency')
    ax6.axvline(np.mean(optimality_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(optimality_scores):.3f}')
    ax6.legend()
    
    plt.tight_layout()
    return fig


def create_simulation_analysis(grid_world, path, total_reward):
    """Create detailed simulation analysis in separate window"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'🚀 Simulation Analysis (Total Reward: {total_reward:.1f})', fontsize=16, fontweight='bold')
    
    # Terrain colors
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    
    # 1. Path Visualization
    ax1 = axes[0, 0]
    ax1.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.7)
    
    if len(path) > 1:
        path_array = np.array(path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.8, label='Path')
        
        # Color-code path by step number
        for i, (row, col) in enumerate(path):
            color_intensity = i / len(path)
            ax1.scatter(col, row, c=[[0, 0, 1, 1-color_intensity]], s=30, edgecolor='white')
        
        # Mark start and end
        ax1.plot(path[0][1], path[0][0], 'go', markersize=12, label='Start')
        ax1.plot(path[-1][1], path[-1][0], 'ro', markersize=12, label='End')
    
    ax1.set_title('Agent Path', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Accumulation
    ax2 = axes[0, 1]
    step_rewards = calculate_step_rewards(grid_world, path)
    cumulative_rewards = np.cumsum(step_rewards)
    
    ax2.plot(range(len(cumulative_rewards)), cumulative_rewards, 'b-', linewidth=2, marker='o')
    ax2.set_title('Cumulative Reward', fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    # Highlight major reward events
    for i, reward in enumerate(step_rewards):
        if abs(reward) > 10:  # Significant reward/penalty
            color = 'green' if reward > 0 else 'red'
            ax2.axvline(i, color=color, alpha=0.3, linestyle='--')
    
    # 3. Step-by-Step Rewards
    ax3 = axes[0, 2]
    colors = ['green' if r > 0 else 'red' if r < -10 else 'blue' for r in step_rewards]
    bars = ax3.bar(range(len(step_rewards)), step_rewards, color=colors, alpha=0.7)
    ax3.set_title('Step-by-Step Rewards', fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Reward')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Exploration Progress
    ax4 = axes[1, 0]
    exploration_progress = calculate_exploration_progress(grid_world, path)
    ax4.plot(range(len(exploration_progress)), exploration_progress, 'g-', linewidth=2, marker='s')
    ax4.set_title('Exploration Progress', fontweight='bold')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Cells Explored')
    ax4.grid(True, alpha=0.3)
    
    # 5. Path Efficiency Analysis
    ax5 = axes[1, 1]
    efficiency_metrics = calculate_path_efficiency(path)
    
    metrics_names = ['Total Steps', 'Unique Cells', 'Backtrack Steps', 'Loop Count']
    metrics_values = [
        len(path),
        len(set(path)),
        efficiency_metrics['backtrack_steps'],
        efficiency_metrics['loop_count']
    ]
    
    bars = ax5.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    ax5.set_title('Path Efficiency Metrics', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(value), ha='center', va='bottom')
    
    # 6. Terrain Encounter Statistics
    ax6 = axes[1, 2]
    terrain_encounters = calculate_terrain_encounters(grid_world, path)
    terrain_labels = ['Unexplored', 'Safe', 'Gas Vent', 'Lava', 'Crater', 'Goal']
    
    valid_encounters = [(label, count) for label, count in zip(terrain_labels, terrain_encounters) if count > 0]
    if valid_encounters:
        labels, counts = zip(*valid_encounters)
        colors_subset = [terrain_colors[i] for i, count in enumerate(terrain_encounters) if count > 0]
        
        wedges, texts, autotexts = ax6.pie(counts, labels=labels, colors=colors_subset, autopct='%1.1f%%')
        ax6.set_title('Terrain Encounters', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_evaluation_dashboard(evaluation_results):
    """Create evaluation dashboard in separate window"""
    if not evaluation_results.evaluation_results:
        return None
        
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('📊 Performance Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    results = evaluation_results.evaluation_results
    
    # 1. Convergence Analysis
    if 'convergence' in results:
        ax1 = axes[0, 0]
        conv = results['convergence']
        deltas = conv['deltas']
        ax1.semilogy(range(len(deltas)), deltas, 'b-', linewidth=2)
        ax1.set_title('Convergence Curve', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Delta (log scale)')
        ax1.grid(True, alpha=0.3)
        
        # Add convergence threshold line
        ax1.axhline(1e-6, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax1.legend()
    
    # 2. Policy Quality Distribution
    if 'policy_quality' in results:
        ax2 = axes[0, 1]
        qual = results['policy_quality']
        rewards = qual['all_rewards']
        
        ax2.hist(rewards, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(qual['avg_reward'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {qual["avg_reward"]:.2f}')
        ax2.set_title('Reward Distribution', fontweight='bold')
        ax2.set_xlabel('Total Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # 3. Gamma Comparison
    if 'gamma_comparison' in results:
        ax3 = axes[1, 0]
        gamma_comp = results['gamma_comparison']
        
        gammas = list(gamma_comp.keys())
        avg_rewards = [gamma_comp[g]['avg_reward'] for g in gammas]
        
        ax3.plot(gammas, avg_rewards, 'bo-', linewidth=2, markersize=8)
        ax3.set_title('Gamma vs Average Reward', fontweight='bold')
        ax3.set_xlabel('Discount Factor (γ)')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Radar Chart
    ax4 = axes[1, 1]
    if 'policy_quality' in results:
        qual = results['policy_quality']
        
        metrics = ['Avg Reward', 'Exploration Rate', 'Safety Score', 'Success Rate']
        values = [
            (qual['avg_reward'] + 50) / 100,  # Normalize to 0-1
            qual['avg_exploration_rate'],
            qual['avg_safety_score'],
            qual['success_rate']
        ]
        
        # Simple radar chart approximation with bar chart
        bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
        ax4.set_title('Performance Metrics', fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 5. Value Function Statistics
    if 'value_function_analysis' in results:
        ax5 = axes[2, 0]
        val_analysis = results['value_function_analysis']
        
        stats = ['Mean', 'Std', 'Min', 'Max', 'Range']
        values = [
            val_analysis['mean_value'],
            val_analysis['std_value'],
            val_analysis['min_value'],
            val_analysis['max_value'],
            val_analysis['value_range']
        ]
        
        bars = ax5.bar(stats, values, color='lightcoral', alpha=0.7)
        ax5.set_title('Value Function Statistics', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Summary Statistics
    ax6 = axes[2, 1]
    ax6.axis('off')  # Hide axes
    
    # Create text summary
    summary_text = "📈 PERFORMANCE SUMMARY\n\n"
    
    if 'convergence' in results:
        conv = results['convergence']
        summary_text += f"🔄 Convergence:\n"
        summary_text += f"  • Iterations: {conv['total_iterations']}\n"
        summary_text += f"  • Time: {conv['total_time']:.2f}s\n\n"
    
    if 'policy_quality' in results:
        qual = results['policy_quality']
        summary_text += f"🎯 Policy Quality:\n"
        summary_text += f"  • Avg Reward: {qual['avg_reward']:.2f}\n"
        summary_text += f"  • Success Rate: {qual['success_rate']:.2%}\n"
        summary_text += f"  • Exploration: {qual['avg_exploration_rate']:.2%}\n"
        summary_text += f"  • Safety: {qual['avg_safety_score']:.2%}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_comparative_analysis(grid_world, value_function, policy):
    """Create comparative analysis in separate window"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🔍 Comparative Analysis', fontsize=16, fontweight='bold')
    
    # 1. Value Function vs Terrain Safety
    ax1 = axes[0, 0]
    safety_map = calculate_terrain_safety(grid_world)
    
    # Flatten for correlation analysis
    values_flat = value_function.flatten()
    safety_flat = safety_map.flatten()
    
    ax1.scatter(safety_flat, values_flat, alpha=0.6, c='blue', s=20)
    ax1.set_xlabel('Terrain Safety Score')
    ax1.set_ylabel('Value Function')
    ax1.set_title('Value vs Terrain Safety', fontweight='bold')
    
    # Add correlation coefficient
    correlation = np.corrcoef(safety_flat, values_flat)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Policy Direction Distribution by Value
    ax2 = axes[0, 1]
    value_bins = np.linspace(np.min(value_function), np.max(value_function), 5)
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    action_by_value = {action: [] for action in range(4)}
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                value = value_function[i, j]
                action = policy[i, j]
                value_bin = np.digitize(value, value_bins) - 1
                action_by_value[action].append(value_bin)
    
    # Create stacked bar chart
    bin_labels = [f'Bin {i+1}' for i in range(len(value_bins)-1)]
    bottom = np.zeros(len(bin_labels))
    
    for action in range(4):
        if action_by_value[action]:
            counts = [action_by_value[action].count(i) for i in range(len(bin_labels))]
            ax2.bar(bin_labels, counts, bottom=bottom, label=action_names[action])
            bottom += counts
    
    ax2.set_title('Action Distribution by Value Range', fontweight='bold')
    ax2.set_xlabel('Value Bins (Low to High)')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    # 3. Exploration Efficiency Heatmap
    ax3 = axes[1, 0]
    exploration_efficiency = calculate_exploration_efficiency(grid_world, value_function)
    im3 = ax3.imshow(exploration_efficiency, cmap='RdYlGn')
    ax3.set_title('Exploration Efficiency', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Efficiency Score')
    
    # 4. Risk vs Reward Analysis
    ax4 = axes[1, 1]
    risk_scores = []
    reward_potential = []
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                risk = calculate_position_risk(grid_world, i, j)
                potential = calculate_reward_potential(grid_world, i, j)
                risk_scores.append(risk)
                reward_potential.append(potential)
    
    # Color points by value function
    values_subset = []
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                values_subset.append(value_function[i, j])
    
    scatter = ax4.scatter(risk_scores, reward_potential, c=values_subset, 
                         cmap='viridis', alpha=0.7, s=30)
    ax4.set_xlabel('Risk Score')
    ax4.set_ylabel('Reward Potential')
    ax4.set_title('Risk vs Reward (colored by Value)', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Value Function')
    
    plt.tight_layout()
    return fig


# Helper functions for enhanced visualizations
def calculate_hazard_proximity(grid_world):
    """Calculate distance to nearest hazard for each cell"""
    hazard_map = np.zeros((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            min_dist = float('inf')
            for hi in range(grid_world.height):
                for hj in range(grid_world.width):
                    if grid_world.original_map[hi, hj] in [2, 3, 4]:  # Hazards
                        dist = abs(i - hi) + abs(j - hj)  # Manhattan distance
                        min_dist = min(min_dist, dist)
            hazard_map[i, j] = min_dist if min_dist != float('inf') else 0
    
    return hazard_map


def calculate_exploration_potential(grid_world):
    """Calculate nearby unexplored cells for each position"""
    potential_map = np.zeros((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            count = 0
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = i + di, j + dj
                    if (0 <= ni < grid_world.height and 0 <= nj < grid_world.width):
                        if grid_world.original_map[ni, nj] == 0:  # Unexplored
                            count += 1
            potential_map[i, j] = count
    
    return potential_map


def calculate_min_hazard_distance(grid_world, row, col):
    """Calculate minimum distance to any hazard"""
    min_dist = float('inf')
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if grid_world.original_map[i, j] in [2, 3, 4]:  # Hazards
                dist = abs(row - i) + abs(col - j)
                min_dist = min(min_dist, dist)
    return min_dist if min_dist != float('inf') else 0


def calculate_policy_consistency(grid_world, policy):
    """Calculate how consistent policy is with neighbors"""
    consistency_map = np.zeros((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j) and policy[i, j] != -1:
                similar_neighbors = 0
                total_neighbors = 0
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < grid_world.height and 0 <= nj < grid_world.width):
                        if not grid_world.is_terminal_state(ni, nj) and policy[ni, nj] != -1:
                            total_neighbors += 1
                            if policy[ni, nj] == policy[i, j]:
                                similar_neighbors += 1
                
                consistency_map[i, j] = similar_neighbors / max(total_neighbors, 1)
    
    return consistency_map


def calculate_policy_entropy(grid_world, policy, value_function):
    """Calculate local policy entropy (diversity)"""
    entropy_map = np.zeros((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                # Get neighbor actions
                neighbor_actions = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < grid_world.height and 0 <= nj < grid_world.width):
                        if not grid_world.is_terminal_state(ni, nj) and policy[ni, nj] != -1:
                            neighbor_actions.append(policy[ni, nj])
                
                if neighbor_actions:
                    # Calculate entropy
                    action_counts = np.bincount(neighbor_actions, minlength=4)
                    probs = action_counts / len(neighbor_actions)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropy_map[i, j] = entropy
    
    return entropy_map


def calculate_action_optimality(grid_world, policy, value_function, row, col):
    """Calculate how optimal the chosen action is"""
    if grid_world.is_terminal_state(row, col) or policy[row, col] == -1:
        return 1.0
    
    # Calculate expected values for all actions
    action_values = []
    for action in range(grid_world.num_actions):
        transitions = grid_world.get_transition_probabilities((row, col), action)
        expected_value = 0
        for prob, next_state, reward in transitions:
            next_row, next_col = next_state
            expected_value += prob * (reward + 0.99 * value_function[next_row, next_col])
        action_values.append(expected_value)
    
    if not action_values:
        return 1.0
    
    chosen_value = action_values[policy[row, col]]
    max_value = max(action_values)
    
    if max_value == min(action_values):
        return 1.0  # All actions equally good
    
    return (chosen_value - min(action_values)) / (max_value - min(action_values))


def calculate_step_rewards(grid_world, path):
    """Calculate reward for each step in the path"""
    if len(path) < 2:
        return []
    
    step_rewards = []
    grid_world.reset_to_original()
    
    for i in range(len(path) - 1):
        current = path[i]
        next_pos = path[i + 1]
        
        # Simulate getting reward for this transition
        reward = grid_world.get_reward(current, 0, next_pos)  # Action doesn't matter for reward calculation
        step_rewards.append(reward)
    
    return step_rewards


def calculate_exploration_progress(grid_world, path):
    """Calculate cumulative exploration progress"""
    explored_cells = set()
    progress = []
    
    for pos in path:
        row, col = pos
        if grid_world.original_map[row, col] == 0:  # Was unexplored
            explored_cells.add(pos)
        progress.append(len(explored_cells))
    
    return progress


def calculate_path_efficiency(path):
    """Calculate various path efficiency metrics"""
    if len(path) < 2:
        return {'backtrack_steps': 0, 'loop_count': 0}
    
    # Count backtracking
    backtrack_steps = 0
    for i in range(len(path) - 1):
        if i > 0 and path[i] == path[i - 1]:
            backtrack_steps += 1
    
    # Count loops (revisiting same cell)
    visited_count = {}
    loop_count = 0
    for pos in path:
        visited_count[pos] = visited_count.get(pos, 0) + 1
        if visited_count[pos] > 1:
            loop_count += 1
    
    return {
        'backtrack_steps': backtrack_steps,
        'loop_count': loop_count
    }


def calculate_terrain_encounters(grid_world, path):
    """Count encounters with each terrain type"""
    encounters = [0] * 6  # For terrain types 0-5 (including goal)
    
    for pos in path:
        row, col = pos
        terrain = grid_world.original_map[row, col]
        if terrain < len(encounters):  # Safety check
            encounters[terrain] += 1
    
    return encounters


def calculate_terrain_safety(grid_world):
    """Calculate safety score for each position"""
    safety_map = np.ones((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            terrain = grid_world.original_map[i, j]
            if terrain == 3 or terrain == 4:  # Lava or Crater
                safety_map[i, j] = 0
            elif terrain == 2:  # Gas Vent
                safety_map[i, j] = 0.3
            elif terrain == 1:  # Safe
                safety_map[i, j] = 1.0
            else:  # Unexplored
                safety_map[i, j] = 0.7
    
    return safety_map


def calculate_exploration_efficiency(grid_world, value_function):
    """Calculate exploration efficiency based on value function and terrain"""
    efficiency_map = np.zeros((grid_world.height, grid_world.width))
    
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if not grid_world.is_terminal_state(i, j):
                # Efficiency = value / (1 + risk)
                risk = 1.0 / (calculate_min_hazard_distance(grid_world, i, j) + 1)
                efficiency_map[i, j] = value_function[i, j] / (1 + risk)
    
    return efficiency_map


def calculate_position_risk(grid_world, row, col):
    """Calculate risk score for a position"""
    min_hazard_dist = calculate_min_hazard_distance(grid_world, row, col)
    return 1.0 / (min_hazard_dist + 1)


def calculate_reward_potential(grid_world, row, col):
    """Calculate reward potential for a position"""
    # Count nearby unexplored cells
    unexplored_nearby = 0
    for di in range(-3, 4):
        for dj in range(-3, 4):
            ni, nj = row + di, col + dj
            if (0 <= ni < grid_world.height and 0 <= nj < grid_world.width):
                if grid_world.original_map[ni, nj] == 0:  # Unexplored
                    distance = abs(di) + abs(dj)
                    unexplored_nearby += 1.0 / (distance + 1)
    
    return unexplored_nearby


def create_final_path_summary(grid_world, value_function, policy, path, total_reward, evaluator=None, start_pos=(0, 0)):
    """
    Create a comprehensive final summary visualization showing the chosen path and results
    
    Args:
        grid_world: VolcanicGridWorld instance
        value_function: Converged value function
        policy: Optimal policy
        path: Simulated agent path
        total_reward: Total reward accumulated
        evaluator: Performance evaluator (optional)
        start_pos: Starting position
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("\n🎯 Generating Final Path Summary Visualization...")
    
    # Create the main summary figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🌋 Volcanic Terrain Exploration - Final Path Summary', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Colors and setup
    terrain_colors = ['lightblue', 'lightgreen', 'orange', 'red', 'darkred', 'gold']
    terrain_cmap = ListedColormap(terrain_colors)
    terrain_labels = ['Unexplored', 'Safe', 'Gas Vent', 'Lava', 'Crater', 'Goal']
    arrow_symbols = ['↑', '↓', '←', '→']
    
    # 1. MAIN PATH VISUALIZATION (Top Left - Large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im1 = ax1.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.8)
    
    # Draw the complete path
    if len(path) > 1:
        path_array = np.array(path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=4, alpha=0.8, label='Agent Path')
        
        # Add path markers
        for i, (row, col) in enumerate(path):
            if i == 0:  # Start position
                ax1.plot(col, row, 'go', markersize=15, markeredgecolor='darkgreen', 
                        markeredgewidth=3, label='Start')
                ax1.text(col, row-0.3, f'START\n{start_pos}', ha='center', va='top', 
                        fontweight='bold', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            elif i == len(path) - 1:  # End position
                ax1.plot(col, row, 'ro', markersize=15, markeredgecolor='darkred', 
                        markeredgewidth=3, label='End')
                ax1.text(col, row+0.4, f'END\n{(row, col)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
            else:  # Intermediate positions
                ax1.plot(col, row, 'yo', markersize=8, alpha=0.7)
    
    # Add policy arrows for context
    for row in range(grid_world.height):
        for col in range(grid_world.width):
            if not grid_world.is_terminal_state(row, col) and policy[row, col] != -1:
                ax1.text(col, row, arrow_symbols[policy[row, col]], 
                        ha="center", va="center", fontsize=12, 
                        color='black', alpha=0.5, fontweight='bold')
    
    ax1.set_title(f'🎯 Final Exploration Path\nSteps: {len(path)-1} | Total Reward: {total_reward:+.1f}', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add terrain legend
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=range(6), shrink=0.8)
    cbar1.set_ticklabels(terrain_labels)
    cbar1.set_label('Terrain Type', fontweight='bold')
    
    # 2. VALUE FUNCTION OVERLAY (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(value_function, cmap='viridis', alpha=0.9)
    
    # Highlight path on value function
    if len(path) > 1:
        path_array = np.array(path)
        ax2.plot(path_array[:, 1], path_array[:, 0], 'w-', linewidth=3, alpha=0.8)
        ax2.plot(path_array[0, 1], path_array[0, 0], 'wo', markersize=10, markeredgecolor='black')
        ax2.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=10, markeredgecolor='white')
    
    ax2.set_title('Value Function\nwith Path Overlay', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. POLICY VISUALIZATION (Middle Right)
    ax3 = fig.add_subplot(gs[1, 2])
    im3 = ax3.imshow(grid_world.original_map, cmap=terrain_cmap, vmin=0, vmax=5, alpha=0.6)
    
    # Draw policy arrows
    for row in range(grid_world.height):
        for col in range(grid_world.width):
            if not grid_world.is_terminal_state(row, col) and policy[row, col] != -1:
                ax3.text(col, row, arrow_symbols[policy[row, col]], 
                        ha="center", va="center", fontsize=10, 
                        color='black', fontweight='bold')
    
    # Highlight path cells
    for row, col in path:
        ax3.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, fill=False, 
                               edgecolor='blue', linewidth=2))
    
    ax3.set_title('Optimal Policy\nwith Path Cells', fontsize=12, fontweight='bold')
    
    # 4. PATH STATISTICS (Top Far Right)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Calculate path statistics
    path_length = len(path) - 1
    unique_cells = len(set(path))
    efficiency = unique_cells / len(path) if len(path) > 0 else 0
    
    # Count terrain encounters
    terrain_encounters = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for row, col in path:
        terrain = grid_world.original_map[row, col]
        if terrain in terrain_encounters:  # Safety check
            terrain_encounters[terrain] += 1
    
    stats_text = f"""📊 PATH ANALYSIS
    
🎯 Movement Statistics:
• Total Steps: {path_length}
• Unique Cells: {unique_cells}
• Path Efficiency: {efficiency:.1%}
• Final Position: {path[-1] if path else 'N/A'}

🌍 Terrain Encounters:
• Unexplored: {terrain_encounters[0]}
• Safe: {terrain_encounters[1]}
• Gas Vents: {terrain_encounters[2]}
• Lava: {terrain_encounters[3]}
• Craters: {terrain_encounters[4]}
• Goal: {terrain_encounters[5]}

💰 Reward Breakdown:
• Total Reward: {total_reward:+.1f}
• Avg per Step: {total_reward/path_length if path_length > 0 else 0:+.1f}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 5. PERFORMANCE METRICS (Middle Far Right)
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis('off')
    
    if evaluator and evaluator.evaluation_results:
        results = evaluator.evaluation_results
        perf_text = f"""🏆 PERFORMANCE SUMMARY
        
📈 Model Performance:
• Success Rate: {results.get('policy_quality', {}).get('success_rate', 0)*100:.1f}%
• Safety Score: {results.get('policy_quality', {}).get('safety_score', 0)*100:.1f}%
• Exploration Rate: {results.get('policy_quality', {}).get('exploration_rate', 0)*100:.1f}%

🧮 Algorithm Results:
• Convergence: {results.get('convergence', {}).get('total_iterations', 'N/A')} iterations
• Solution Time: {results.get('convergence', {}).get('total_time', 0):.2f}s

⚖️ Best Gamma: {results.get('gamma_comparison', {}).get('best_gamma', 0.99)}
"""
    else:
        perf_text = f"""🏆 PERFORMANCE SUMMARY
        
📈 Current Run:
• Path Length: {path_length} steps
• Efficiency: {efficiency:.1%}
• Final Reward: {total_reward:+.1f}

🎯 Status: Single Run Analysis
(Run comprehensive evaluation 
for detailed statistics)
"""
    
    ax5.text(0.05, 0.95, perf_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # 6. REWARD PROGRESSION (Bottom - Full Width)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate cumulative rewards along path
    cumulative_rewards = [0]
    current_reward = 0
    
    for i in range(1, len(path)):
        prev_pos = path[i-1]
        curr_pos = path[i]
        # Get a fresh grid world to calculate proper rewards
        temp_grid = grid_world.original_map.copy()
        if temp_grid[curr_pos[0], curr_pos[1]] == 0:  # Unexplored
            step_reward = 20 - 1  # Exploration reward minus living cost
        elif temp_grid[curr_pos[0], curr_pos[1]] == 1:  # Safe
            step_reward = -1  # Living cost
        elif temp_grid[curr_pos[0], curr_pos[1]] == 2:  # Gas vent
            step_reward = -50 - 1  # Penalty plus living cost
        elif temp_grid[curr_pos[0], curr_pos[1]] == 3:  # Lava
            step_reward = -1000 - 1  # Terminal penalty
        elif temp_grid[curr_pos[0], curr_pos[1]] == 4:  # Crater
            step_reward = -1000 - 1  # Terminal penalty
        else:
            step_reward = -1  # Default living cost
            
        current_reward += step_reward
        cumulative_rewards.append(current_reward)
    
    steps = list(range(len(cumulative_rewards)))
    ax6.plot(steps, cumulative_rewards, 'b-', linewidth=3, marker='o', markersize=6)
    ax6.fill_between(steps, cumulative_rewards, alpha=0.3)
    
    # Add reward annotations for significant changes
    for i, reward in enumerate(cumulative_rewards):
        if i == 0 or i == len(cumulative_rewards) - 1:
            ax6.annotate(f'{reward:+.0f}', (i, reward), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
    
    ax6.set_xlabel('Step Number', fontweight='bold')
    ax6.set_ylabel('Cumulative Reward', fontweight='bold')
    ax6.set_title(f'🎁 Reward Progression Along Path (Final: {total_reward:+.1f})', 
                 fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Add final summary box
    final_summary = f"🏁 MISSION SUMMARY: Explored {unique_cells} cells in {path_length} steps with {total_reward:+.1f} total reward"
    fig.text(0.5, 0.02, final_summary, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save the summary if needed
    try:
        plt.savefig('volcanic_exploration_final_summary.png', dpi=300, bbox_inches='tight')
        print("📸 Final summary saved as 'volcanic_exploration_final_summary.png'")
    except Exception as e:
        print(f"Note: Could not save summary image: {e}")
    
    plt.show()
    
    return fig




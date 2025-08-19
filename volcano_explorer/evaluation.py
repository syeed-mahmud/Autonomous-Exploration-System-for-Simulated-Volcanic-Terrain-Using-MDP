"""
Model Evaluation Module

This module provides comprehensive evaluation metrics and analysis
for the MDP-based volcanic terrain exploration system.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
try:
    from .solver import value_iteration, extract_policy
except ImportError:
    from solver import value_iteration, extract_policy


class PerformanceEvaluator:
    """
    Evaluates the performance of the MDP solver and policies
    """
    
    def __init__(self, grid_world):
        """
        Initialize the evaluator with a grid world
        
        Args:
            grid_world: VolcanicGridWorld instance
        """
        self.grid_world = grid_world
        self.evaluation_results = {}
        
    def evaluate_convergence(self, gamma=0.99, theta=1e-6):
        """
        Evaluate convergence properties of value iteration
        
        Returns:
            Dictionary with convergence metrics
        """
        print("🔍 Evaluating Value Iteration Convergence...")
        
        # Track convergence
        V = np.zeros((self.grid_world.height, self.grid_world.width))
        iterations = []
        deltas = []
        computation_times = []
        
        iteration = 0
        start_time = time.time()
        
        while True:
            iter_start = time.time()
            delta = 0
            V_old = V.copy()
            
            # Update value for each state
            for row in range(self.grid_world.height):
                for col in range(self.grid_world.width):
                    state = (row, col)
                    
                    if self.grid_world.is_terminal_state(row, col):
                        continue
                    
                    action_values = []
                    for action in range(self.grid_world.num_actions):
                        transitions = self.grid_world.get_transition_probabilities(state, action)
                        expected_value = 0
                        for prob, next_state, reward in transitions:
                            next_row, next_col = next_state
                            expected_value += prob * (reward + gamma * V_old[next_row, next_col])
                        action_values.append(expected_value)
                    
                    if action_values:
                        V[row, col] = max(action_values)
                        delta = max(delta, abs(V[row, col] - V_old[row, col]))
            
            iteration += 1
            iter_time = time.time() - iter_start
            
            iterations.append(iteration)
            deltas.append(delta)
            computation_times.append(iter_time)
            
            if delta < theta:
                break
        
        total_time = time.time() - start_time
        
        convergence_metrics = {
            'total_iterations': iteration,
            'total_time': total_time,
            'avg_time_per_iteration': total_time / iteration,
            'final_delta': delta,
            'convergence_rate': self._calculate_convergence_rate(deltas),
            'iterations': iterations,
            'deltas': deltas,
            'computation_times': computation_times,
            'final_value_function': V
        }
        
        self.evaluation_results['convergence'] = convergence_metrics
        return convergence_metrics
    
    def evaluate_policy_quality(self, policy, num_simulations=100):
        """
        Evaluate the quality of a policy through multiple simulations
        
        Args:
            policy: Policy to evaluate
            num_simulations: Number of simulation runs
            
        Returns:
            Dictionary with policy quality metrics
        """
        print("📊 Evaluating Policy Quality...")
        
        rewards = []
        path_lengths = []
        exploration_rates = []
        safety_scores = []
        
        for sim in range(num_simulations):
            # Reset environment for each simulation
            self.grid_world.reset_to_original()
            
            # Random starting position (avoid terminal states)
            valid_starts = []
            for i in range(self.grid_world.height):
                for j in range(self.grid_world.width):
                    if not self.grid_world.is_terminal_state(i, j):
                        valid_starts.append((i, j))
            
            start_pos = valid_starts[np.random.randint(len(valid_starts))]
            
            # Simulate path
            path, total_reward = self._simulate_with_policy(policy, start_pos)
            
            # Calculate metrics
            rewards.append(total_reward)
            path_lengths.append(len(path))
            
            # Exploration rate: unique unexplored cells visited
            explored_cells = self._count_explored_cells(path)
            total_unexplored = np.sum(self.grid_world.original_map == 0)
            exploration_rate = explored_cells / max(total_unexplored, 1)
            exploration_rates.append(exploration_rate)
            
            # Safety score: ratio of safe moves to total moves
            safety_score = self._calculate_safety_score(path)
            safety_scores.append(safety_score)
        
        quality_metrics = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_path_length': np.mean(path_lengths),
            'std_path_length': np.std(path_lengths),
            'avg_exploration_rate': np.mean(exploration_rates),
            'avg_safety_score': np.mean(safety_scores),
            'success_rate': np.mean([r > 0 for r in rewards]),
            'all_rewards': rewards,
            'all_path_lengths': path_lengths,
            'all_exploration_rates': exploration_rates,
            'all_safety_scores': safety_scores
        }
        
        self.evaluation_results['policy_quality'] = quality_metrics
        return quality_metrics
    
    def compare_gamma_values(self, gamma_values=[0.9, 0.95, 0.99, 0.995]):
        """
        Compare performance across different discount factors
        
        Args:
            gamma_values: List of gamma values to compare
            
        Returns:
            Dictionary with comparison results
        """
        print("⚖️ Comparing Different Gamma Values...")
        
        comparison_results = {}
        
        for gamma in gamma_values:
            print(f"  Testing γ = {gamma}")
            
            # Solve with this gamma
            V = value_iteration(self.grid_world, gamma=gamma, theta=1e-6, verbose=False)
            policy = extract_policy(self.grid_world, V, gamma=gamma)
            
            # Quick policy evaluation (fewer simulations for speed)
            quality = self.evaluate_policy_quality(policy, num_simulations=20)
            
            comparison_results[gamma] = {
                'avg_reward': quality['avg_reward'],
                'avg_exploration_rate': quality['avg_exploration_rate'],
                'avg_safety_score': quality['avg_safety_score'],
                'success_rate': quality['success_rate']
            }
        
        self.evaluation_results['gamma_comparison'] = comparison_results
        return comparison_results
    
    def analyze_value_function_properties(self, value_function):
        """
        Analyze statistical properties of the value function
        
        Args:
            value_function: 2D numpy array of values
            
        Returns:
            Dictionary with value function analysis
        """
        print("📈 Analyzing Value Function Properties...")
        
        # Flatten non-terminal values
        non_terminal_values = []
        for i in range(self.grid_world.height):
            for j in range(self.grid_world.width):
                if not self.grid_world.is_terminal_state(i, j):
                    non_terminal_values.append(value_function[i, j])
        
        non_terminal_values = np.array(non_terminal_values)
        
        analysis = {
            'mean_value': np.mean(non_terminal_values),
            'std_value': np.std(non_terminal_values),
            'min_value': np.min(non_terminal_values),
            'max_value': np.max(non_terminal_values),
            'value_range': np.max(non_terminal_values) - np.min(non_terminal_values),
            'value_distribution': non_terminal_values,
            'high_value_cells': np.sum(non_terminal_values > np.mean(non_terminal_values)),
            'negative_value_cells': np.sum(non_terminal_values < 0)
        }
        
        self.evaluation_results['value_function_analysis'] = analysis
        return analysis
    
    def _simulate_with_policy(self, policy, start_pos, max_steps=100):
        """Simulate agent following policy from start position"""
        path = [start_pos]
        total_reward = 0
        current_state = start_pos
        
        for step in range(max_steps):
            row, col = current_state
            
            if self.grid_world.is_terminal_state(row, col):
                break
            
            action = policy[row, col]
            if action == -1:
                break
            
            next_state = self.grid_world.get_next_state(current_state, action)
            reward = self.grid_world.get_reward(current_state, action, next_state)
            
            total_reward += reward
            current_state = next_state
            path.append(current_state)
            
            if len(path) > 1 and path[-1] == path[-2]:
                break
        
        return path, total_reward
    
    def _count_explored_cells(self, path):
        """Count unique unexplored cells visited in path"""
        explored = set()
        for pos in path:
            row, col = pos
            if self.grid_world.original_map[row, col] == 0:  # Was unexplored
                explored.add(pos)
        return len(explored)
    
    def _calculate_safety_score(self, path):
        """Calculate safety score based on hazard encounters"""
        hazard_encounters = 0
        for pos in path:
            row, col = pos
            terrain = self.grid_world.original_map[row, col]
            if terrain in [2, 3, 4]:  # Gas, Lava, Crater
                hazard_encounters += 1
        
        return max(0, 1 - hazard_encounters / len(path))
    
    def _calculate_convergence_rate(self, deltas):
        """Estimate convergence rate from delta values"""
        if len(deltas) < 10:
            return 0
        
        # Linear regression on log(delta) vs iteration
        log_deltas = np.log(np.array(deltas[-100:]) + 1e-10)  # Avoid log(0)
        iterations = np.arange(len(log_deltas))
        
        if len(log_deltas) > 1:
            slope = np.polyfit(iterations, log_deltas, 1)[0]
            return -slope  # Negative slope indicates convergence
        return 0
    
    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report"""
        if not self.evaluation_results:
            print("❌ No evaluation results available. Run evaluations first.")
            return
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Convergence Analysis
        if 'convergence' in self.evaluation_results:
            conv = self.evaluation_results['convergence']
            print(f"\n🔄 CONVERGENCE ANALYSIS:")
            print(f"  • Total Iterations: {conv['total_iterations']}")
            print(f"  • Total Time: {conv['total_time']:.2f} seconds")
            print(f"  • Avg Time/Iteration: {conv['avg_time_per_iteration']:.4f} seconds")
            print(f"  • Convergence Rate: {conv['convergence_rate']:.6f}")
            print(f"  • Final Delta: {conv['final_delta']:.2e}")
        
        # Policy Quality
        if 'policy_quality' in self.evaluation_results:
            qual = self.evaluation_results['policy_quality']
            print(f"\n🎯 POLICY QUALITY ANALYSIS:")
            print(f"  • Average Reward: {qual['avg_reward']:.2f} ± {qual['std_reward']:.2f}")
            print(f"  • Reward Range: [{qual['min_reward']:.1f}, {qual['max_reward']:.1f}]")
            print(f"  • Average Path Length: {qual['avg_path_length']:.1f} ± {qual['std_path_length']:.1f}")
            print(f"  • Exploration Rate: {qual['avg_exploration_rate']:.2%}")
            print(f"  • Safety Score: {qual['avg_safety_score']:.2%}")
            print(f"  • Success Rate: {qual['success_rate']:.2%}")
        
        # Gamma Comparison
        if 'gamma_comparison' in self.evaluation_results:
            gamma_comp = self.evaluation_results['gamma_comparison']
            print(f"\n⚖️ GAMMA COMPARISON:")
            for gamma, metrics in gamma_comp.items():
                print(f"  • γ = {gamma}: Reward={metrics['avg_reward']:.1f}, "
                      f"Exploration={metrics['avg_exploration_rate']:.2%}, "
                      f"Safety={metrics['avg_safety_score']:.2%}")
        
        # Value Function Analysis
        if 'value_function_analysis' in self.evaluation_results:
            val_analysis = self.evaluation_results['value_function_analysis']
            print(f"\n📈 VALUE FUNCTION ANALYSIS:")
            print(f"  • Mean Value: {val_analysis['mean_value']:.2f}")
            print(f"  • Value Range: {val_analysis['value_range']:.2f}")
            print(f"  • High-Value Cells: {val_analysis['high_value_cells']}")
            print(f"  • Negative-Value Cells: {val_analysis['negative_value_cells']}")
        
        print("\n" + "="*60)


def run_comprehensive_evaluation(grid_world, policy=None, value_function=None):
    """
    Run a comprehensive evaluation of the MDP system
    
    Args:
        grid_world: VolcanicGridWorld instance
        policy: Optional pre-computed policy
        value_function: Optional pre-computed value function
        
    Returns:
        PerformanceEvaluator instance with results
    """
    evaluator = PerformanceEvaluator(grid_world)
    
    # If no policy/value function provided, compute them
    if value_function is None:
        print("Computing value function for evaluation...")
        convergence_metrics = evaluator.evaluate_convergence()
        value_function = convergence_metrics['final_value_function']
    
    if policy is None:
        print("Extracting policy for evaluation...")
        policy = extract_policy(grid_world, value_function)
    
    # Run all evaluations
    evaluator.evaluate_policy_quality(policy, num_simulations=50)
    evaluator.compare_gamma_values()
    evaluator.analyze_value_function_properties(value_function)
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report()
    
    return evaluator

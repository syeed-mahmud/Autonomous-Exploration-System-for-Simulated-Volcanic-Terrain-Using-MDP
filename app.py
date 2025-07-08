import numpy as np
import matplotlib.pyplot as plt
import random
from enum import Enum
from typing import Tuple, List, Dict

# ========================================
# MDP TUPLE DEFINITION
# ========================================

class TerrainType(Enum):
    """States representing different terrain types"""
    SAFE = 0
    LAVA = 1
    GAS = 2
    CRATER = 3
    GOAL = 4

class Action(Enum):
    """Action space for the MDP"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class VolcanicMDP:
    """
    MDP Formulation for Volcanic Terrain Exploration
    
    MDP Tuple: (S, A, T, R, γ)
    - S: State space (grid positions with terrain types)
    - A: Action space (movement directions)
    - T: Transition function P(s'|s,a)
    - R: Reward function R(s,a,s')
    - γ: Discount factor
    """
    
    def __init__(self, grid_size: int = 5, discount_factor: float = 0.9):
        self.grid_size = grid_size
        self.discount_factor = discount_factor  # γ (gamma)
        
        # Initialize MDP components
        self.states = self._define_state_space()
        self.actions = list(Action)
        self.terrain_grid = self._initialize_terrain()
        
        print("🌋 MDP FORMULATION FOR VOLCANIC TERRAIN EXPLORATION")
        print("=" * 55)
        print(f"State Space (S): {len(self.states)} states")
        print(f"Action Space (A): {len(self.actions)} actions")
        print(f"Discount Factor (γ): {self.discount_factor}")
        print("=" * 55)
        
    def _define_state_space(self) -> List[Tuple[int, int]]:
        """
        Define state space S
        Each state represents a position (x, y) in the grid
        """
        states = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                states.append((x, y))
        return states
    
    def _initialize_terrain(self) -> np.ndarray:
        """Initialize terrain with different hazard types"""
        terrain = np.full((self.grid_size, self.grid_size), TerrainType.SAFE.value)
        
        # Place hazards randomly
        num_hazards = max(1, (self.grid_size * self.grid_size) // 4)
        hazard_positions = random.sample(self.states[1:-1], min(num_hazards, len(self.states)-2))
        
        for pos in hazard_positions:
            hazard_type = random.choice([TerrainType.LAVA, TerrainType.GAS, TerrainType.CRATER])
            terrain[pos] = hazard_type.value
        
        # Set goal position
        terrain[self.grid_size-1, self.grid_size-1] = TerrainType.GOAL.value
        
        return terrain
    
    def transition_function(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """
        Transition Function T: P(s'|s,a)
        Deterministic transition (probability = 1.0)
        Returns next state given current state and action
        """
        x, y = state
        
        if action == Action.NORTH:
            next_state = (max(0, x - 1), y)
        elif action == Action.SOUTH:
            next_state = (min(self.grid_size - 1, x + 1), y)
        elif action == Action.EAST:
            next_state = (x, min(self.grid_size - 1, y + 1))
        elif action == Action.WEST:
            next_state = (x, max(0, y - 1))
        else:
            next_state = state  # Invalid action, stay in place
            
        return next_state
    
    def reward_function(self, state: Tuple[int, int], action: Action, next_state: Tuple[int, int]) -> float:
        """
        Reward Function R(s,a,s')
        Returns immediate reward for transitioning from state to next_state via action
        """
        terrain_type = self.terrain_grid[next_state]
        
        # Define rewards based on terrain type
        if terrain_type == TerrainType.GOAL.value:
            return 100.0  # High reward for reaching goal
        elif terrain_type == TerrainType.LAVA.value:
            return -100.0  # High penalty for lava
        elif terrain_type == TerrainType.GAS.value:
            return -50.0   # Medium penalty for gas
        elif terrain_type == TerrainType.CRATER.value:
            return -75.0   # High penalty for crater
        elif terrain_type == TerrainType.SAFE.value:
            return -1.0    # Small cost for movement
        else:
            return -10.0   # Default penalty
    
    def display_mdp_components(self):
        """Display the MDP components clearly"""
        print("\n📊 MDP COMPONENTS:")
        print("-" * 30)
        
        print("1. STATE SPACE (S):")
        print(f"   • Grid positions: {self.grid_size}×{self.grid_size} = {len(self.states)} states")
        print(f"   • State representation: (x, y) coordinates")
        print(f"   • Example states: {self.states[:5]}...")
        
        print("\n2. ACTION SPACE (A):")
        for action in self.actions:
            print(f"   • {action.name}: Move {action.name.lower()}")
        
        print("\n3. TRANSITION FUNCTION T(s'|s,a):")
        print("   • Deterministic transitions")
        print("   • Boundary constraints (cannot move outside grid)")
        print("   • Example: State (1,1) + Action NORTH → State (0,1)")
        
        print("\n4. REWARD FUNCTION R(s,a,s'):")
        print("   • Goal state: +100")
        print("   • Lava: -100")
        print("   • Gas emission: -50") 
        print("   • Crater: -75")
        print("   • Safe terrain: -1")
        
        print(f"\n5. DISCOUNT FACTOR (γ): {self.discount_factor}")
        print("   • Controls importance of future rewards")

class SimplePolicy:
    """
    Policy Definition π(a|s)
    Maps states to actions
    """
    
    def __init__(self, mdp: VolcanicMDP):
        self.mdp = mdp
        self.policy_table = {}
        self._initialize_policy()
        
    def _initialize_policy(self):
        """Initialize with a simple heuristic policy"""
        print("\n🎯 POLICY DEFINITION:")
        print("-" * 25)
        print("Policy π(a|s): Maps each state to an action")
        
        for state in self.mdp.states:
            # Simple heuristic: move towards goal (bottom-right corner)
            x, y = state
            goal_x, goal_y = self.mdp.grid_size - 1, self.mdp.grid_size - 1
            
            # Choose action based on distance to goal
            if x < goal_x and y < goal_y:
                # Randomly choose between SOUTH or EAST
                action = random.choice([Action.SOUTH, Action.EAST])
            elif x < goal_x:
                action = Action.SOUTH
            elif y < goal_y:
                action = Action.EAST
            else:
                action = Action.SOUTH  # Default
                
            self.policy_table[state] = action
    
    def get_action(self, state: Tuple[int, int]) -> Action:
        """Get action for given state according to policy"""
        return self.policy_table.get(state, Action.SOUTH)
    
    def display_policy(self):
        """Display the policy in a readable format"""
        print("\n📋 POLICY TABLE π(a|s):")
        print("-" * 35)
        print("State (x,y) → Action")
        print("-" * 35)
        
        # Show first few states as examples
        for i, (state, action) in enumerate(list(self.policy_table.items())[:10]):
            print(f"({state[0]},{state[1]})      → {action.name}")
        
        if len(self.policy_table) > 10:
            print(f"... and {len(self.policy_table) - 10} more states")

def demonstrate_mdp_model():
    """Demonstrate the MDP model and policy"""
    
    # Create MDP
    mdp = VolcanicMDP(grid_size=5, discount_factor=0.9)
    
    # Display MDP components
    mdp.display_mdp_components()
    
    # Create and display policy
    policy = SimplePolicy(mdp)
    policy.display_policy()
    
    # Demonstrate model usage
    print("\n🚀 MODEL DEMONSTRATION:")
    print("-" * 30)
    
    # Example state transitions
    start_state = (0, 0)
    print(f"Starting at state: {start_state}")
    
    for step in range(5):
        action = policy.get_action(start_state)
        next_state = mdp.transition_function(start_state, action)
        reward = mdp.reward_function(start_state, action, next_state)
        
        print(f"Step {step + 1}: {start_state} --{action.name}--> {next_state} (Reward: {reward})")
        start_state = next_state
        
        # Stop if reached goal or hazard
        terrain_type = mdp.terrain_grid[next_state]
        if terrain_type != TerrainType.SAFE.value:
            terrain_name = TerrainType(terrain_type).name
            print(f"Reached {terrain_name} terrain. Stopping demonstration.")
            break
    
    # Display terrain grid
    print("\n🗺️ TERRAIN GRID:")
    print("-" * 20)
    terrain_symbols = {
        TerrainType.SAFE.value: '.',
        TerrainType.LAVA.value: 'L',
        TerrainType.GAS.value: 'G',
        TerrainType.CRATER.value: 'C',
        TerrainType.GOAL.value: 'X'
    }
    
    for x in range(mdp.grid_size):
        row = ""
        for y in range(mdp.grid_size):
            terrain_type = mdp.terrain_grid[x, y]
            symbol = terrain_symbols.get(terrain_type, '?')
            row += f"{symbol} "
        print(row)
    
    print("\nLegend: . = Safe, L = Lava, G = Gas, C = Crater, X = Goal")
    
    return mdp, policy

if __name__ == "__main__":
    print("CSE440 - Group 5 - Volcanic Terrain MDP")
    print("Mid-Update: Basic MDP Formulation")
    print()
    
    mdp, policy = demonstrate_mdp_model()
    


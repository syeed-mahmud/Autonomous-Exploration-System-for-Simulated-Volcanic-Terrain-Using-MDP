# 🌋 Autonomous Exploration System for Volcanic Terrain Using MDP

**CSE440 - Artificial Intelligence**  
**Group 5 - Mid Update**  
**Section:** 1  
**Faculty:** MSRB  

##  Project Overview

This project formulates an autonomous exploration system for volcanic terrain navigation as a **Markov Decision Process (MDP)**. The mid-update focuses on defining the core MDP components and establishing the mathematical foundation for decision-making under uncertainty.

##  MDP Formulation

Our system is modeled as a 5-tuple MDP: **(S, A, T, R, γ)**

### 1. State Space (S)
- **Definition**: Set of all possible positions in the volcanic terrain grid
- **Representation**: (x, y) coordinates where agent can be located
- **Size**: Grid_size × Grid_size positions
- **Example**: For 5×5 grid → 25 possible states

### 2. Action Space (A)
- **NORTH**: Move up (decrease x-coordinate)
- **SOUTH**: Move down (increase x-coordinate)  
- **EAST**: Move right (increase y-coordinate)
- **WEST**: Move left (decrease y-coordinate)

### 3. Transition Function T(s'|s,a)
- **Type**: Deterministic transitions
- **Rule**: Agent moves to adjacent cell unless blocked by boundary
- **Boundary Handling**: Agent stays in current position if action leads outside grid
- **Mathematical Form**: P(s'|s,a) = 1 for valid transitions, 0 otherwise

### 4. Reward Function R(s,a,s')
Immediate rewards based on destination terrain type:
- **Goal State**: +100 (mission success)
- **Safe Terrain**: -1 (movement cost)
- **Lava Flow**: -100 (high danger penalty)
- **Gas Emission**: -50 (moderate danger penalty)
- **Crater**: -75 (high danger penalty)

### 5. Discount Factor (γ)
- **Value**: 0.9
- **Purpose**: Balance immediate vs. future rewards
- **Effect**: Encourages finding paths to goal while considering long-term consequences

##  Policy Definition

### Policy π(a|s)
- **Type**: Deterministic policy mapping states to actions
- **Implementation**: Heuristic-based initial policy
- **Strategy**: Move towards goal position while avoiding hazards
- **Representation**: Policy table storing action for each state

### Heuristic Policy Rules:
1. If both x and y coordinates are less than goal coordinates → Choose SOUTH or EAST randomly
2. If only x coordinate is less than goal → Choose SOUTH
3. If only y coordinate is less than goal → Choose EAST
4. Default action → SOUTH

##  Implementation Structure

### Core Components:

**`VolcanicMDP` Class**
- Defines state space and action space
- Implements transition function T(s'|s,a)
- Implements reward function R(s,a,s')
- Manages terrain grid with hazards

**`SimplePolicy` Class**  
- Implements policy π(a|s)
- Provides action selection for any given state
- Uses heuristic approach for initial policy

##  Quick Start

### Prerequisites
- Python 3.7+
- NumPy, Matplotlib

### Installation
```bash
cd "Group 5"
python app.py
```

### Expected Output
The program will display:
1. MDP tuple components (S, A, T, R, γ)
2. Policy definition π(a|s)
3. Example state transitions
4. Terrain grid visualization

##  Example Execution

```
 MDP FORMULATION FOR VOLCANIC TERRAIN EXPLORATION
=======================================================
State Space (S): 25 states
Action Space (A): 4 actions  
Discount Factor (γ): 0.9
=======================================================

 MDP COMPONENTS:
1. STATE SPACE (S): Grid positions: 5×5 = 25 states
2. ACTION SPACE (A): NORTH, SOUTH, EAST, WEST
3. TRANSITION FUNCTION T(s'|s,a): Deterministic transitions
4. REWARD FUNCTION R(s,a,s'): Based on terrain type
5. DISCOUNT FACTOR (γ): 0.9

 POLICY DEFINITION:
Policy π(a|s): Maps each state to an action

 MODEL DEMONSTRATION:
Step 1: (0,0) --SOUTH--> (1,0) (Reward: -1)
Step 2: (1,0) --EAST--> (1,1) (Reward: -1)
...
```

##  Next Steps (Final Update)

### Planned Enhancements:
1. **Value Iteration Algorithm**: Compute optimal policy π*
2. **Q-Learning Implementation**: Learn optimal policy through experience
3. **Dynamic Environment**: Add probabilistic hazard changes
4. **Safety Constraints**: Implement risk-aware exploration
5. **Performance Evaluation**: Compare different policies and algorithms

### Advanced Features:
- **Stochastic Transitions**: Add uncertainty to movement
- **Partial Observability**: Limited visibility of terrain
- **Multi-Agent Coordination**: Multiple explorers working together
- **Deep Reinforcement Learning**: Neural network-based policies

## 💡 Key Insights

### MDP Advantages for This Problem:
1. **Uncertainty Modeling**: Handles unpredictable volcanic environment
2. **Sequential Decision Making**: Plans multi-step exploration strategies  
3. **Reward Optimization**: Balances exploration efficiency with safety
4. **Mathematical Foundation**: Provides formal framework for analysis

### Design Decisions:
- **Deterministic Transitions**: Simplifies initial model development
- **Grid-Based States**: Discrete state space for tractable computation
- **Immediate Rewards**: Clear feedback for policy learning
- **Heuristic Policy**: Reasonable starting point for optimization

##  Team Information

**Group 5 - CSE440 Section 1**
- MDP formulation and mathematical modeling
- Policy definition and implementation
- System architecture design



**Current Status**: Mid-Update - MDP Formulation Complete  
**Next Milestone**: Final Update - Algorithm Implementation and Evaluation 
# POMCP Model

import numpy as np
import random
from collections import defaultdict

class POMCPNode:
    def __init__(self):
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Estimated value of the node
        self.children = {}  # Dictionary storing child nodes for actions

class POMCP:
    def __init__(self, actions, gamma=0.95, uct_c=1.0, num_simulations=100):
        self.actions = actions  # List of possible actions
        self.gamma = gamma  # Discount factor for future rewards
        self.uct_c = uct_c  # Exploration constant for UCT
        self.num_simulations = num_simulations  # Number of simulations to run
        self.tree = {}  # Dictionary representing the search tree
    
    def search(self, belief, env, horizon=10):
        """Runs POMCP simulations to find the best action given the belief."""
        for _ in range(self.num_simulations):
            state = random.choice(belief)  # Sample a state from belief particles
            self.simulate(state, env, horizon)
        
        return self.best_action(belief)
    
    def simulate(self, state, env, depth):
        """Simulates a POMDP trajectory from a given state to estimate action values."""
        if depth == 0:
            return 0  # End recursion at max depth
        
        key = tuple(state)  # Convert state to a tuple for dictionary storage
        if key not in self.tree:
            self.tree[key] = POMCPNode()
            return self.rollout(state, env, depth)  # Perform rollout if state is new
        
        node = self.tree[key]
        action = self.select_action(node)
        next_state, reward, observation = env.step(state, action)
        
        if (key, action, observation) not in self.tree:
            self.tree[(key, action, observation)] = POMCPNode()
        
        # Recursively simulate the next state and update the value estimate
        q_value = reward + self.gamma * self.simulate(next_state, env, depth - 1)
        
        # Update node statistics
        node.visits += 1
        node.value += (q_value - node.value) / node.visits
        
        return q_value
    
    def select_action(self, node):
        """Selects an action using Upper Confidence Bound for Trees (UCT)."""
        best_value = -float('inf')
        best_action = None
        
        for action in self.actions:
            if action not in node.children:
                return action  # Try unexplored actions first
            
            q_value = node.children[action].value
            uct_value = q_value + self.uct_c * np.sqrt(np.log(node.visits + 1) / (node.children[action].visits + 1e-6))
            
            if uct_value > best_value:
                best_value = uct_value
                best_action = action
        
        return best_action
    
    def rollout(self, state, env, depth):
        """Performs a random rollout to estimate future rewards."""
        if depth == 0:
            return 0
        action = random.choice(self.actions)  # Random action selection
        next_state, reward, _ = env.step(state, action)
        return reward + self.gamma * self.rollout(next_state, env, depth - 1)
    
    def best_action(self, belief):
        """Selects the best action based on estimated action values from the tree."""
        action_values = defaultdict(float)
        action_visits = defaultdict(int)
        
        for state in belief:
            key = tuple(state)
            if key in self.tree:
                for action in self.actions:
                    if action in self.tree[key].children:
                        action_values[action] += self.tree[key].children[action].value
                        action_visits[action] += self.tree[key].children[action].visits
        
        # Normalize action values by visit count
        for action in action_values:
            if action_visits[action] > 0:
                action_values[action] /= action_visits[action]
        
        return max(action_values, key=action_values.get, default=random.choice(self.actions))

### For Reference
class TigerEnv:
    """The Tiger Problem environment for testing POMCP."""
    def __init__(self):
        self.states = ['tiger-left', 'tiger-right']
    
    def reset(self):
        """Resets the environment by randomly placing the tiger behind a door."""
        return random.choice(self.states)
    
    def step(self, state, action):
        """Executes an action and returns the next state, reward, and observation."""
        if action == 'listen':
            observation = state if random.random() < 0.85 else self.states[1 - self.states.index(state)]
            return state, -1, observation  # Listening has a small cost
        if action == 'open-left':
            return state, 10 if state == 'tiger-right' else -100, None  # Reward or penalty
        if action == 'open-right':
            return state, 10 if state == 'tiger-left' else -100, None

class ISSEnv:
    """The International Space Station environment"""
    def __init__(self):
        self.states = ['blocked_berth1', 'blocked_berth2','blocked_berth3']
    
    def reset(self):
        """Resets the environment by randomly blocking one of the berths."""
        return random.choice(self.states)
    
    def step(self, state, action):
        """Executes an action and returns the next state, reward, and observation."""

        ## bin_drop_berth# -> would ask the user for help with deciding where to place the bag
        
        if action == 'bin_drop_berth1':
            observation = state if random.random() < 0.85 else self.states[1 - self.states.index(state)] # TODO: Update this line
            return state, -2, observation  # asking for help has a small cost
        if action == 'bin_drop_berth2':
            observation = state if random.random() < 0.85 else self.states[1 - self.states.index(state)] # TODO: Update this line
            return state, -2, observation  # asking for help has a small cost
        if action == 'bin_drop_berth3':
            observation = state if random.random() < 0.85 else self.states[1 - self.states.index(state)] # TODO: Update this line
            return state, -2, observation  # asking for help has a small cost
        if action == 'silent_drop_berth1':
            return state, -100 if state == 'blocked_berth1' else 10, None  # penalty or reward
        if action == 'silent_drop_berth2':
            return state, -100 if state == 'blocked_berth2' else 10, None  # penalty or reward
        if action == 'silent_drop_berth3':
            return state, -100 if state == 'blocked_berth3' else 10, None
        
# Running POMCP on the Tiger Problem
env = ISSEnv()
#       bin_berth_#  -> binary question (yes/no)
pomcp = POMCP(actions=['silent_drop_berth1', 'silent_drop_berth2', 'silent_drop_berth3','bin_drop_berth1','bin_drop_berth2','bin_drop_berth3'])
initial_belief = ['blocked_berth1', 'blocked_berth2','blocked_berth3'] * 33.3  # Initial belief as particles

for _ in range(10):  # Run 10 trials
    action = pomcp.search(initial_belief, env)
    print(f'Chosen Action: {action}')
print(f'')

# Notes
# if berth 1 is blocked best option is to place in berth 2
# else the robot will be given a small penalty for placing in berth 3 which is farther

# Can have a problem where after several iterations, it may create an unwanted pattern
# careful with data manipulation, may lose data or create noise

# handle duplicate data and large data set

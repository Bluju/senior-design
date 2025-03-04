# POMDP Model

import numpy as np

class POMDP:
    def __init__(self, states, actions, observations, transition_model, observation_model, reward_model, discount):
        """Initialize the POMDP with states, actions, observations, and models."""
        self.states = states  # List of possible states
        self.actions = actions  # List of possible actions
        self.observations = observations  # List of possible observations
        self.T = transition_model  # Transition model P(s' | s, a)
        self.O = observation_model  # Observation model P(o | s')
        self.R = reward_model  # Reward function R(s, a)
        self.discount = discount  # Discount factor for future rewards
        self.belief = np.ones(len(states)) / len(states)  # Start with uniform belief over states
    
    def update_belief(self, action, observation):
        """Update the belief state using Bayesian filtering based on action and received observation."""
        new_belief = np.zeros(len(self.states))
        for s_prime in range(len(self.states)):
            total = 0
            for s in range(len(self.states)):
                total += self.belief[s] * self.T[s, action, s_prime]  # Apply transition model
            new_belief[s_prime] = self.O[s_prime, action, observation] * total  # Apply observation model
        new_belief /= np.sum(new_belief)  # Normalize the belief to maintain probability distribution
        self.belief = new_belief  # Update belief state
    
    def lookahead_search(self, horizon):
        """Perform look-ahead search to determine the best action based on expected utility."""
        best_action = None
        best_utility = -np.inf
        for action in range(len(self.actions)):
            utility = self.simulate_action(action, self.belief, horizon, {})
            if utility > best_utility:
                best_utility = utility
                best_action = action
        return best_action  # Return the action that maximizes expected utility
    
    def simulate_action(self, action, belief, horizon, cache):
        """Simulate an action's expected utility over a given horizon using memoization."""
        key = (tuple(belief), action, horizon)
        if key in cache:
            return cache[key]  # Return precomputed utility if available
        
        if horizon == 0:
            return np.dot(belief, self.R[:, action])  # Compute immediate reward expectation
        
        expected_utility = 0
        for s_prime in range(len(self.states)):
            next_belief = np.zeros(len(self.states))
            for s in range(len(self.states)):
                next_belief[s_prime] += belief[s] * self.T[s, action, s_prime]  # Apply transition model
            next_belief /= np.sum(next_belief) if np.sum(next_belief) > 0 else 1  # Normalize belief
            best_future_action = self.lookahead_search(horizon - 1)  # Recursive call for next decision step
            expected_utility += np.dot(next_belief, self.R[:, best_future_action])  # Compute future reward
        
        cache[key] = expected_utility  # Store result in cache
        return expected_utility

# Example Problem: Tiger Problem
states = ["Tiger-Left", "Tiger-Right"]  # Possible locations of the tiger
actions = ["Listen", "Open-Left", "Open-Right"]  # Possible actions the agent can take
observations = ["Hear-Left", "Hear-Right"]  # Possible observations the agent can receive
discount = 0.9  # Discount factor for future rewards

# Transition Model (stays the same unless opened, then resets)
T = np.array([[[1, 0], [0.5, 0.5], [0.5, 0.5]],  # If starting from Tiger-Left
              [[0, 1], [0.5, 0.5], [0.5, 0.5]]]) # If starting from Tiger-Right

# Observation Model (uncertain hearing)
O = np.array([[[0.85, 0.15], [0.85, 0.15], [0.85, 0.15]],  # If tiger is left
              [[0.15, 0.85], [0.15, 0.85], [0.15, 0.85]]]) # If tiger is right

# Reward Model
R = np.array([[-1, -100, 10],  # If tiger is left, listening has small cost, opening wrong door is bad
              [-1, 10, -100]]) # If tiger is right, opening the correct door gives reward

# Create a POMDP instance
pomdp = POMDP(states, actions, observations, T, O, R, discount)

# Run an example decision-making process
for _ in range(5):
    action = pomdp.lookahead_search(horizon=5)  # Use look-ahead search to decide the best action
    print(f"Chosen action: {actions[action]}")
    observation = np.random.choice(len(observations), p=O[:, action, 0])  # Sample an observation
    pomdp.update_belief(action, observation)  # Update belief based on the new observation
    print(f"Updated belief: {pomdp.belief}")  # Print the updated belief state


# Notes
# if berth 1 is blocked best option is to place in berth 2
# else the robot will be given a small penalty for placing in berth 3 which is farther

# Can have a problem where after several iterations, it may create an unwanted pattern
# careful with data manipulation, may lose data or create noise

# handle duplicate data and large data set

#TODO: incorporate Situational Awareness and Workload Estimation
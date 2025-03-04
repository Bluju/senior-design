# POMDP Model

import numpy as np

class POMDP:
    def __init__(self, states, actions, observations, transition_model, observation_model, reward_model, discount):
        self.states = states  # Possible states
        self.actions = actions  # Possible actions
        self.observations = observations  # Possible observations
        self.T = transition_model  # Transition model P(s' | s, a)
        self.O = observation_model  # Observation model P(o | s')
        self.R = reward_model  # Reward function R(s, a)
        self.discount = discount  # Discount factor
        self.belief = np.ones(len(states)) / len(states)  # Uniform initial belief
    
    def update_belief(self, action, observation):
        """Bayesian Filtering: Update belief state based on action and observation."""
        new_belief = np.zeros(len(self.states))
        for s_prime in range(len(self.states)):
            total = 0
            for s in range(len(self.states)):
                total += self.belief[s] * self.T[s, action, s_prime]
            new_belief[s_prime] = self.O[s_prime, action, observation] * total
        new_belief /= np.sum(new_belief)  # Normalize
        self.belief = new_belief
    
    def lookahead_search(self, horizon):
        """Look-ahead search to find the best action based on utility."""
        best_action = None
        best_utility = -np.inf
        for action in range(len(self.actions)):
            utility = self.simulate_action(action, self.belief, horizon)
            if utility > best_utility:
                best_utility = utility
                best_action = action
        return best_action
    
    def simulate_action(self, action, belief, horizon):
        """Project forward the expected utility of an action sequence."""
        if horizon == 0:
            return np.dot(belief, self.R[:, action])
        expected_utility = 0
        for s_prime in range(len(self.states)):
            next_belief = np.zeros(len(self.states))
            for s in range(len(self.states)):
                next_belief[s_prime] += belief[s] * self.T[s, action, s_prime]
            next_belief /= np.sum(next_belief) if np.sum(next_belief) > 0 else 1
            best_future_action = self.lookahead_search(horizon - 1)
            expected_utility += np.dot(next_belief, self.R[:, best_future_action])
        return expected_utility

# Example Problem: Tiger Problem
states = ["Tiger-Left", "Tiger-Right"]
actions = ["Listen", "Open-Left", "Open-Right"]
observations = ["Hear-Left", "Hear-Right"]
discount = 0.9

# Transition Model (stays the same unless opened, then resets)
T = np.array([[[1, 0], [0.5, 0.5], [0.5, 0.5]],  # From Tiger-Left
              [[0, 1], [0.5, 0.5], [0.5, 0.5]]]) # From Tiger-Right

# Observation Model (uncertain hearing)
O = np.array([[[0.85, 0.15], [0.85, 0.15], [0.85, 0.15]],  # If tiger is left
              [[0.15, 0.85], [0.15, 0.85], [0.15, 0.85]]]) # If tiger is right

# Reward Model
R = np.array([[-1, -100, 10],  # If tiger is left
              [-1, 10, -100]]) # If tiger is right

pomdp = POMDP(states, actions, observations, T, O, R, discount)

# Run an example decision-making process
for _ in range(5):
    action = pomdp.lookahead_search(horizon=2)
    print(f"Chosen action: {actions[action]}")
    observation = np.random.choice(len(observations), p=O[:, action, 0])
    pomdp.update_belief(action, observation)


# Notes
# if berth 1 is blocked best option is to place in berth 2
# else the robot will be given a small penalty for placing in berth 3 which is farther

# Can have a problem where after several iterations, it may create an unwanted pattern
# careful with data manipulation, may lose data or create noise

# handle duplicate data and large data set

#TODO: incorporate Situational Awareness and Workload Estimation
# POMDP Model using the International Space Station Problem
# Note that this will be a simplified version of the complete problem

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
        
        # Exclude "Listen" if at the last step of the horizon
        available_actions = range(len(self.actions))
        
        for action in available_actions:
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

# Simplified Problem: Tiger Problem
## states are written as situationalAwareness_workload_block_berth_number
states = ["low_low","high_high","low_high","high_low"]  # SituationalAwareness_Workload pairs
actions = ["silent", "wh-question"]  # Communication strategies to choose from
observations = ["blocked_berth1", "blocked_berth2"]  # Possible observations the agent can receive
discount = 0.9  # Discount factor for future rewards

# Transition Model (stays the same unless opened, then resets)
# Structure: T[s,a,s']
# where s(row)    -> current state (before taking action)
#       a(column) -> action taken
#       s'(depth) -> next state (after action)
# so for each state (s) we define how each action (a) affects the probability of ending up in each next state (s')
# Our transition model needs to be a 4 x 2 x 2 array 

#                silent    wh_question
T = np.array([[[0.5, 0.5], [0.5, 0.5]],  # low_low     #low SA, low WL
              [[0.5, 0.5], [0.5, 0.5]],  # high_high
              [[0.6, 0.4], [0.6, 0.4]],  # low_high    #low SA, high WL   (maybe more likely to stay silent)
              [[0.5, 0.5], [0.5, 0.5]]]) # high_low
#               

# Observation Model 
# TODO: I don't think what I have is right. 
# TODO: Are these the probability of choosing the communication strategy or probability of berth being blocked? 
# action picked: silent       wh_question
O = np.array([[[0.60, 0.40], [0.40, 0.60]],  # low_low
              [[0.80, 0.20], [0.40, 0.60]],  # high_high
              [[0.80, 0.20], [0.70, 0.30]],  # low_high
              [[0.30, 0.70], [0.30, 0.70]]]) # high_low

# TODO: Silent communication will have uncertainty in the choice 
# TODO: Wh-question communication strategy will be very likely of being correct 
# Reward Model
R = np.array([[-100, 10],  # If berth1 is blocked 
              [10, -100]]) # If berth2 is blocked

# Create a POMDP instance
pomdp = POMDP(states, actions, observations, T, O, R, discount)

# Run an example decision-making process
for _ in range(5):
    action = pomdp.lookahead_search(horizon=5)  # Use look-ahead search to decide the best action
    print(f"Chosen action: {actions[action]}\taction: {action}")
    if action != 0:  # If the action is not "Listen"
        print(f"Reward: {np.dot(pomdp.belief, R[:, action])}")  # Print the expected reward
        break # Stop the process if the action is to open a door
    
    observation = np.random.choice(len(observations), p=O[:, action, 0])  # Sample an observation
    print(f"Observation: {observations[observation]}")
    pomdp.update_belief(action, observation)  # Update belief based on the new observation
    print(f"Updated belief: {pomdp.belief}")  # Print the updated belief state


# Notes
# The horizon variable is used to determine the depth of the look-ahead search
# The simulate_action method uses memoization to avoid redundant calculations


# if berth 1 is blocked best option is to place in berth 2
# else the robot will be given a small penalty for placing in berth 3 which is farther

# Can have a problem where after several iterations, it may create an unwanted pattern
# careful with data manipulation, may lose data or create noise

# handle duplicate data and large data set

#TODO: incorporate Situational Awareness and Workload Estimation

# For testing, we can assume that a high situational awareness and low workload contribute towards greater accuracy in decision making


# simplify the problem
# start with two berths
# two workloads
# two situational awareness

# give the model a state (sa, w)
# then the model will decide which communcation strategy to use

# output the confidence in all the actions
# example: updated belief: [0.75 0.10 0.5]
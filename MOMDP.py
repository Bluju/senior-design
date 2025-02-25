import pandas as pd
import numpy as np

class MOMDP:
    def __init__(self, csv_file):
        self.data = self.load_data(csv_file)
        self.states = self.define_states()
        self.actions = ['silent', 'yes_no', 'wh_question']
        self.observations = ['low', 'medium', 'high']
        self.transition_probs = self.initialize_transitions()
        self.observation_probs = self.initialize_observations()
        self.rewards = self.initialize_rewards()
    
    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        return df
    
    def define_states(self):
        return list(self.data[['workload_level', 'autonomy_strategy']].drop_duplicates().itertuples(index=False, name=None))
    
    def initialize_transitions(self):
        transitions = {}
        for action in self.actions:
            transitions[action] = np.random.rand(len(self.states), len(self.states))
            transitions[action] /= transitions[action].sum(axis=1, keepdims=True)  # Normalize
        return transitions
    
    def initialize_observations(self):
        observations = {}
        for action in self.actions:
            observations[action] = np.random.rand(len(self.states), len(self.observations))
            observations[action] /= observations[action].sum(axis=1, keepdims=True)
        return observations
    
    def initialize_rewards(self):
        rewards = {}
        for action in self.actions:
            rewards[action] = np.random.rand(len(self.states))
        return rewards
    
    def get_next_state(self, current_state, action):
        return np.random.choice(self.states, p=self.transition_probs[action][self.states.index(current_state)])
    
    def get_observation(self, current_state, action):
        return np.random.choice(self.observations, p=self.observation_probs[action][self.states.index(current_state)])
    
    def get_reward(self, current_state, action):
        return self.rewards[action][self.states.index(current_state)]
    
# Example usage
momdp_model = MOMDP('data.csv')
print("States:", momdp_model.states)
print("Sample Transition Probabilities:", momdp_model.transition_probs['silent'])

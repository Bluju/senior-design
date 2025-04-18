import pandas as pd
import numpy as np

class POMDPValueIterationSolver:
    def __init__(self, df, gamma=0.9, threshold=1e-3):
        """
        Implements a POMDP solver with belief updates and value iteration.
        :param df: DataFrame containing (state, action, observation, end_state, total_reward).
        :param gamma: Discount factor for future rewards.
        :param threshold: Convergence threshold for value iteration.

        The solver computes the optimal policy using value iteration and provides confidence scores
        based on the belief of the current state and the transition certainty.

        The belief update is performed using Bayes' rule, and the transition probabilities are
        computed from the dataset.

        The value iteration process is used to compute the optimal policy for the POMDP.

        1. Load the csv file and extract relevant columns from the survey results.
        2. After counting the frequency of each state-action pair, the transition probabilities are computed.
        3. Uses the Bayes' rule to update the belief states based on the action and observation.
        4. We then use the Bellman equation to compute the optimal policy and value function.
        5. In the output, a confidence score is provided based on the belief of the state and the frequency of observed transitions


        """
        self.df = df.dropna()
        self.gamma = gamma
        self.threshold = threshold

        # Extract unique states (both start and end states), actions, and observations
        start_states = set(self.df["state"].unique())
        end_states = set(self.df["end_state"].unique())
        self.states = sorted(start_states.union(end_states))  # Include all states

        self.actions = sorted(self.df["action"].unique())
        self.observations = sorted(self.df["observation"].unique())

        # Initialize rewards, transitions, and beliefs
        self.rewards = {row["end_state"]: row["total_reward"] for _, row in self.df.iterrows()}
        self.transition_counts = {}
        self.transition_probs = {}
        self.beliefs = {state: 1 / len(self.states) for state in self.states}  # Uniform initial belief

        self._compute_transitions()

        # Initialize value function and policy for all states
        self.value_function = {s: 0 for s in self.states}
        self.policy = {s: None for s in self.states}

        self.value_iteration()

    def _compute_transitions(self):
        """
        Compute transition probabilities from the dataset.
        """
        state_action_counts = {}

        for _, row in self.df.iterrows():
            key = (row["state"], row["action"], row["observation"])
            if key not in self.transition_counts:
                self.transition_counts[key] = {}
                state_action_counts[(row["state"], row["action"])] = 0

            end_state = row["end_state"]
            if end_state not in self.transition_counts[key]:
                self.transition_counts[key][end_state] = 0

            self.transition_counts[key][end_state] += 1
            state_action_counts[(row["state"], row["action"])] += 1

        # Normalize transition probabilities
        for key, transitions in self.transition_counts.items():
            total = sum(transitions.values())
            self.transition_probs[key] = {s: count / total for s, count in transitions.items()}

        # Store duplicate frequencies for confidence calculation
        self.state_action_counts = state_action_counts

    def belief_update(self, action, observation):
        """
        Update belief states using Bayes' rule.
        """
        new_beliefs = {}
        normalization_factor = 0

        for s_prime in self.states:
            sum_prob = 0
            for s in self.states:
                key = (s, action, observation)
                transition_prob = self.transition_probs.get(key, {}).get(s_prime, 0)
                sum_prob += transition_prob * self.beliefs[s]

            new_beliefs[s_prime] = sum_prob
            normalization_factor += sum_prob

        # Normalize beliefs
        for s in self.states:
            if normalization_factor > 0:
                new_beliefs[s] /= normalization_factor
            else:
                new_beliefs[s] = 1 / len(self.states)

        self.beliefs = new_beliefs

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal policy.
        """
        while True:
            delta = 0
            new_value_function = self.value_function.copy()

            for state in self.states:
                best_value = float('-inf')
                best_action = None

                for action in self.actions:
                    expected_value = 0

                    for obs in self.observations:
                        key = (state, action, obs)
                        if key in self.transition_probs:
                            for end_state, prob in self.transition_probs[key].items():
                                reward = self.rewards.get(end_state, 0)
                                expected_value += prob * (reward + self.gamma * self.value_function.get(end_state, 0))

                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action

                new_value_function[state] = best_value
                self.policy[state] = best_action
                delta = max(delta, abs(best_value - self.value_function[state]))

            self.value_function = new_value_function

            if delta < self.threshold:
                break

    def get_optimal_action(self, current_state):
        """
        Retrieve the optimal action and a more refined confidence score for a given state.
        """
        optimal_action = self.policy.get(current_state, None)

        # Compute confidence based on the belief of the state and the transition certainty
        belief_confidence = self.beliefs.get(current_state, 0)

        # Adjust confidence by factoring in action transition frequency
        action_confidences = []
        for action in self.actions:
            key = (current_state, action)
            action_occurrences = self.state_action_counts.get(key, 0)

            if action_occurrences > 0:
                # Use log-based weighting to avoid over-scaling
                action_confidences.append(np.log1p(action_occurrences))

        if action_confidences:
            action_confidence = sum(action_confidences) / len(action_confidences)  # Average confidence
            confidence = min(1.0, belief_confidence * action_confidence)  # Normalize
        else:
            confidence = belief_confidence

        return optimal_action, round(confidence, 2)


# Load the datasets
file_path_nonmal = "create_nonmal_NoDuplicates.csv"
file_path_mal = "create_mal_NoDuplicates.csv"

df_nonmal = pd.read_csv(file_path_nonmal)
df_mal = pd.read_csv(file_path_mal)

# Extract relevant columns
relevant_columns = df_nonmal.columns[9:13].tolist() + [df_nonmal.columns[-1]]
df_nonmal = df_nonmal[relevant_columns]
df_mal = df_mal[relevant_columns]

# Rename columns for clarity
df_nonmal.columns = ["state", "action", "observation", "end_state", "total_reward"]
df_mal.columns = ["state", "action", "observation", "end_state", "total_reward"]

# Instantiate the improved solver
solver_nonmal_vi = POMDPValueIterationSolver(df_nonmal)
solver_mal_vi = POMDPValueIterationSolver(df_mal)

# Test with multiple states
test_states = df_mal["state"].dropna().unique()[:]  # Select :n sample states for testing
results = []

for state in test_states:
    optimal_action, confidence_score = solver_mal_vi.get_optimal_action(state)
    results.append((state, optimal_action, confidence_score))

# Convert results into a DataFrame for better visualization
print(f'\nNon-malfunction data: {file_path_nonmal}\nMalfunction data: {file_path_mal}\n')
df_results = pd.DataFrame(results, columns=["State", "Optimal Action", "Confidence Score"])
print(df_results)
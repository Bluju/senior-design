import random
import copy
import pandas as pd

DEBUG = True  # Set to True for detailed debug output

# ============================================
# 1. Excel Data Integration and Aggregation
# ============================================
def parse_data(file):
    """
    Reads an Excel file with no header and extracts:
      - state (column 10, index 9)
      - action (column 11, index 10)
      - observation (column 12, index 11)
      - end_state (column 13, index 12)
      - reward (column 17, index 16)
    Also extracts the blocked berth from the state string and determines if the outcome is valid.
    """
    df = pd.read_csv(file, header=None)
    df_sub = df.iloc[:, [9, 10, 11, 12, 16]].copy()
    df_sub.columns = ['state', 'action', 'observation', 'end_state', 'reward']
    # Extract blocked berth from the state string (e.g., "LowSA_LowWL_blocked_berth1" -> "berth1")
    df_sub['blocked_berth'] = df_sub['state'].apply(lambda s: s.split("_")[-1] if isinstance(s, str) else None)
    # Define outcome validity: if reward is -100 then outcome is invalid, otherwise valid.
    df_sub['valid'] = df_sub['reward'] != -100
    return df_sub

def compute_transition_probabilities(normal_file, malfunction_file):
    df_normal = parse_data(normal_file)
    df_malfunction = parse_data(malfunction_file)

    def aggregate(df):
        agg = {}
        grouped = df.groupby(["action", "blocked_berth"])["valid"].agg(['mean', 'count'])
        for index, row in grouped.iterrows():
            action, blocked_berth = index
            valid_fraction = row['mean']

            # Laplace smoothing: Adjust probability calculations
            smoothing_factor = 10  
            smoothed_valid = (valid_fraction * row['count'] + smoothing_factor) / (row['count'] + (2 * smoothing_factor))
            smoothed_invalid = 1 - smoothed_valid

            agg[(action, blocked_berth)] = {"valid": smoothed_valid, "invalid": smoothed_invalid}
        return agg

    aggregated_normal = aggregate(df_normal)
    aggregated_malfunction = aggregate(df_malfunction)

    if DEBUG:
        print("Aggregated probabilities (Normal Data):")
        for key, prob in aggregated_normal.items():
            print(f"  {key}: {prob}")
        print("\nAggregated probabilities (Malfunction Data):")
        for key, prob in aggregated_malfunction.items():
            print(f"  {key}: {prob}")

    return aggregated_normal, aggregated_malfunction

# ============================================
# 2. State, Actions, and Observations
# ============================================
class State:
    def __init__(self, workload, blocked_berth, drop_berth, valid, SA, waiting, malfunction):
        # workload: 0=low, 1=medium, 2=high
        # blocked_berth: which berth is obstructed; e.g. "berth1", "berth2", "berth3"
        # drop_berth: the berth where the drop occurs (e.g., "drop_berth1")
        # valid: Boolean flag indicating if the outcome is valid
        # SA: situation awareness (0=low, 1=high)
        # waiting: Boolean flag if the robot is in a waiting state
        # malfunction: Boolean flag; True if sensor malfunctioning
        self.workload = workload
        self.blocked_berth = blocked_berth
        self.drop_berth = drop_berth
        self.valid = valid
        self.SA = SA
        self.waiting = waiting
        self.malfunction = malfunction

    def __repr__(self):
        return (f"State(workload={self.workload}, blocked_berth={self.blocked_berth}, "
                f"drop_berth={self.drop_berth}, valid={self.valid}, SA={self.SA}, "
                f"waiting={self.waiting}, malfunction={self.malfunction})")

# Define action sets.
physical_actions = ["silent_drop_berth1", "silent_drop_berth2", "silent_drop_berth3"]
comm_actions = ["comm_yesNoQuestion_drop_berth1", "comm_yesNoQuestion_drop_berth2",
                "comm_yesNoQuestion_drop_berth3", "comm_whQuestion"]
ACTIONS = physical_actions + comm_actions
OBSERVATIONS = ["No_ob_comm", "command none", "Command_drop_berth1", "Command_drop_berth2", "Command_drop_berth3"]

# ============================================
# 3. Reward Function
# ============================================
def reward_function(state, action, next_state):
    if not next_state.valid:
        return -100
    r = 0
    if next_state.drop_berth is not None:
        if "berth3" in next_state.drop_berth:
            r = 10 - 2
        else:
            r = 10
    if action.startswith("comm"):
        r -= 1  # additional cost for communication actions
    return r

# ============================================
# 4. Transition Model (using aggregated Excel data)
# ============================================
# These will be loaded as global variables.
aggregated_normal = {}
aggregated_malfunction = {}

def transition_model(state, action):
    new_state = copy.deepcopy(state)
    new_state.drop_berth = None
    new_state.valid = False

    # Determine which aggregated probabilities to use based on malfunction flag.
    if state.malfunction:
        agg = aggregated_malfunction
    else:
        agg = aggregated_normal

    # For physical actions.
    if action in physical_actions:
        intended_berth = action.split("_")[-1]  # e.g., "berth1"
        key = (action, state.blocked_berth)
        if key in agg:
            outcome = random.choices(["valid", "invalid"],
                                     weights=[agg[key]["valid"], agg[key]["invalid"]],
                                     k=1)[0]
            if outcome == "valid":
                new_state.drop_berth = "drop_" + intended_berth
                new_state.valid = True
            else:
                new_state.valid = False
        else:
            # Fallback: original logic if no data exists.
            if not state.malfunction:
                if intended_berth == state.blocked_berth:
                    new_state.valid = False
                else:
                    new_state.drop_berth = "drop_" + intended_berth
                    new_state.valid = True
            else:
                p_correct = 0.3
                if random.random() < p_correct:
                    if intended_berth == state.blocked_berth:
                        new_state.valid = False
                    else:
                        new_state.drop_berth = "drop_" + intended_berth
                        new_state.valid = True
                else:
                    if intended_berth == state.blocked_berth:
                        new_state.drop_berth = "drop_" + intended_berth
                        new_state.valid = True
                    else:
                        new_state.valid = False

    # For yes/no communication actions.
    elif action.startswith("comm_yesNoQuestion"):
        target = action.split("_")[-1]  # e.g., "drop_berth1"
        key = (action, state.blocked_berth)
        if key in agg:
            outcome = random.choices(["valid", "invalid"],
                                     weights=[agg[key]["valid"], agg[key]["invalid"]],
                                     k=1)[0]
            if outcome == "valid":
                new_state.drop_berth = target
                new_state.valid = True
            else:
                new_state.valid = False
        else:
            if not state.malfunction:
                if target.replace("drop_", "") == state.blocked_berth:
                    new_state.valid = False
                else:
                    if random.random() < 0.8:
                        new_state.drop_berth = target
                        new_state.valid = True
                    else:
                        new_state.valid = False
            else:
                if target.replace("drop_", "") == state.blocked_berth:
                    if random.random() < 0.5:
                        new_state.drop_berth = target
                        new_state.valid = True
                    else:
                        new_state.valid = False
                else:
                    if random.random() < 0.5:
                        new_state.drop_berth = target
                        new_state.valid = True
                    else:
                        new_state.valid = False

    # For wh question actions.
    elif action == "comm_whQuestion":
        key = (action, state.blocked_berth)
        if key in agg:
            outcome = random.choices(["valid", "invalid"],
                                     weights=[agg[key]["valid"], agg[key]["invalid"]],
                                     k=1)[0]
            if outcome == "valid":
                chosen = random.choice(["drop_berth1", "drop_berth2", "drop_berth3"])
                new_state.drop_berth = chosen
                new_state.valid = True
            else:
                new_state.valid = False
        else:
            valid_prob = 0.7 if not state.malfunction else 0.4
            if random.random() < valid_prob:
                chosen = random.choice(["drop_berth1", "drop_berth2", "drop_berth3"])
                if chosen.replace("drop_", "") == state.blocked_berth:
                    new_state.valid = False
                else:
                    new_state.drop_berth = chosen
                    new_state.valid = True
            else:
                new_state.valid = False

    return new_state

# ============================================
# 5. Observation Model
# ============================================
def observation_model(state, action, next_state):
    if next_state.valid and next_state.drop_berth:
        if next_state.drop_berth == "drop_berth1":
            return "Command_drop_berth1"
        elif next_state.drop_berth == "drop_berth2":
            return "Command_drop_berth2"
        elif next_state.drop_berth == "drop_berth3":
            return "Command_drop_berth3"
    return "command none"

# ============================================
# 6. Belief Update (Particle Filter)
# ============================================
def update_belief(belief_particles, action, observation, num_particles=100):
    new_particles = []
    while len(new_particles) < num_particles:
        particle = random.choice(belief_particles)
        next_state = transition_model(particle, action)
        obs = observation_model(particle, action, next_state)
        if obs == observation:
            new_particles.append(next_state)
    if len(new_particles) == 0:
        new_particles = belief_particles
    return new_particles

# ============================================
# 7. POMCP Search (Monte Carlo Tree Search)
# ============================================
def pomcp_search(belief_particles, depth, max_depth=10, action_counts=None, state_action_counts=None):
    if depth == max_depth:
        return 0  # Stop recursion at max depth

    state = random.choice(belief_particles)
    best_value = -float('inf')
    best_action = None

    # Initialize tracking dictionaries if not provided
    if action_counts is None:
        action_counts = {action: 0 for action in ACTIONS}

    if state_action_counts is None:
        state_action_counts = {}

    if depth == 0 and DEBUG:
        print("\n--- POMCP Root Level Action Details ---")

    for action in ACTIONS:
        next_state = transition_model(state, action)
        r = reward_function(state, action, next_state)
        obs = observation_model(state, action, next_state)
        new_belief = update_belief(belief_particles, action, obs, num_particles=50)

        # Track state-action visit count
        state_action_key = (state.blocked_berth, action)
        state_action_counts[state_action_key] = state_action_counts.get(state_action_key, 0) + 1

        future_value = pomcp_search(new_belief, depth + 1, max_depth, action_counts, state_action_counts)

        visit_count = state_action_counts[state_action_key]
        alpha = 0.1  # Learning rate for updating rewards
        total_value = max((1 - alpha) * best_value + alpha * (r + future_value), -50)  # Clamp values

        if depth == 0 and DEBUG:
            print(f"Action: {action}, Immediate Reward: {r}, Future Value: {future_value}, Visit Count: {visit_count}")
            print(f"Total Value (Backpropagated & Smoothed): {total_value}\n")

        if total_value > best_value:
            best_value = total_value
            best_action = action

    if depth == 0:
        if best_action is None:
            print("Warning: No valid action found, selecting fallback action.")
            best_action = random.choice(ACTIONS)

        action_counts[best_action] += 1  # Track best action selection

        # Compute confidence score with duplicate weighting
        total_selections = sum(action_counts.values())
        decay_factor = 0.05  # Reduce overconfidence for repeated selections
        confidence_penalty = decay_factor * (sum(state_action_counts.values()) / (total_selections + 1))

        confidence_score = (action_counts[best_action] / total_selections) * (1 - confidence_penalty)

        # Clamp confidence score between 0.05 and 1 (never 0)
        confidence_score = max(0.05, min(confidence_score, 1))

        print(f"--- Best Action at Root: {best_action} with Total Value: {best_value} ---")
        print(f"--- Confidence Score for '{best_action}': {confidence_score:.2f} (Adjusted for Duplicates) ---\n")

        return best_action, confidence_score

    return best_value

# ============================================
# 8. Data Deduplication (if needed)
# ============================================
def deduplicate_data(data):
    dedup = {}
    for record in data:
        key = (record['start_state'], record['action'], record['observation'], record['end_state'])
        if key not in dedup:
            dedup[key] = {'count': 0, 'reward': 0}
        dedup[key]['count'] += 1
        dedup[key]['reward'] += record['reward']
    for key in dedup:
        dedup[key]['avg_reward'] = dedup[key]['reward'] / dedup[key]['count']
    return dedup

# ============================================
# 9. Main Execution
# ============================================
if __name__ == "__main__":
    # Load the aggregated probabilities from your Excel files.
    # Replace the filenames below with the actual paths to your Excel files.
    aggregated_normal, aggregated_malfunction = compute_transition_probabilities("create_nonmal.csv",
                                                                                "create_mal.csv")
    
    # Create an initial belief with 100 particles.
    initial_belief = []
    malfunction_flag = True  # Set to True for malfunction experiment; change to False for normal.
    for _ in range(100):
        state = State(
            workload=random.choice([0, 1, 2]),
            blocked_berth=random.choice(["berth1", "berth2", "berth3"]),
            drop_berth=None,
            valid=True,
            SA=random.choice([0, 1]),
            waiting=False,
            malfunction=malfunction_flag
        )
        initial_belief.append(state)
    
    # Print initial belief summary.
    belief_summary = {}
    for s in initial_belief:
        key = (s.blocked_berth, s.malfunction)
        belief_summary[key] = belief_summary.get(key, 0) + 1
    if DEBUG:
        print("Initial Belief Distribution (blocked_berth, malfunction):")
        for key, count in belief_summary.items():
            print(f"  {key}: {count} particles")
    
    # Simulate an action and observation. INSERT ANY ACTION AND OBSERVATION HERE FOR TESTING AND OUTPUT PURPOSES
    action = "silent_drop_berth1"
    observation = "command none"  # In practice, this comes from the operator. <- This part isn't needed, needs to be investigated.
    
    # Update the belief based on the action and observation.
    updated_belief = update_belief(initial_belief, action, observation)
    updated_summary = {}
    for s in updated_belief:
        key = (s.blocked_berth, s.malfunction, s.valid, s.drop_berth)
        updated_summary[key] = updated_summary.get(key, 0) + 1
    if DEBUG:
        print("\nUpdated Belief Distribution after action '{}' and observation '{}':".format(action, observation))
        for key, count in updated_summary.items():
            print(f"  {key}: {count} particles")
    
    # Run POMCP search to determine the best next action.
    best_action = pomcp_search(updated_belief, depth=0, max_depth=5)
    print("Final Recommended Action:", best_action)

    #TODO: Update the code so that it shows the confidence level of the recommended action.
    #TODO: Provide an explanation of how the tree is updating the values of the nodes.
    #TODO: Take a look at the aggregation of the probabilities and see if it can be improved (Might be errors due to the binary values seen in the aggregated probabilities).
    #TODO: When training the model, make sure the duplicate values are considered as well so that the confidence score will be affected.
    #TODO: Understand how the backpropogation of the algorithm works to update the values of the rewards and the nodes.
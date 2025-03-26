POMDP Solver for Robot Item Placement in Berths
Overview
This project implements a Partially Observable Markov Decision Process (POMDP) solver for a robot placing items into berths. The robot operates under uncertainty, meaning it may not always have complete knowledge of the environment, such as whether a berth is blocked. The algorithm uses Monte Carlo Tree Search (POMCP) to estimate the best action given its belief about the world.

The POMDP solver utilizes:

Excel survey data for learning transition probabilities.
Particle filtering for belief state updates.
Monte Carlo Planning (POMCP) for decision-making.
1. Algorithm Explanation
1.1 POMDP Model Components
A POMDP is defined as a tuple (S, A, T, O, R, γ) where:

S (States): Represents the environment, including:

The blocked berth (e.g., "berth1", "berth2", "berth3").
Whether the robot’s sensor is malfunctioning.
The workload level (low, medium, high).
The situation awareness (SA) level.
The berth where the item was dropped.
Whether the action taken resulted in a valid outcome.
A (Actions): The robot can either:

Physically drop an item in a berth (silent actions).
Communicate with the operator through yes/no or wh-questions.
T (Transition Model): Defines the probability of moving from one state to another after taking an action.

The transition model is learned from Excel data, meaning it uses empirical probabilities rather than fixed values.
O (Observations): The robot receives an observation after performing an action. These observations help refine the belief state.

Example observations: "Command_drop_berth1", "Command_drop_berth2", "Command_drop_berth3", "command none".
R (Reward Function): The goal is to maximize reward:

A valid drop gives +10 points.
Dropping into berth3 has a -2 penalty.
Asking a communication question has a -1 penalty.
An invalid drop results in -100 points.
γ (Discount Factor): Encourages the robot to plan for the future rather than only focusing on immediate rewards.

1.2 Particle Filter for Belief Updates
Since the robot cannot observe the true state directly, it maintains a belief state—a probability distribution over possible states.

A particle filter is used to update the belief.
The belief state is represented by a set of sampled states (particles).
After taking an action and receiving an observation, the particles are updated:
The robot samples a particle from the current belief.
It simulates what would happen if that state was real.
If the simulated observation matches the actual observation, the particle is kept.
If not enough particles match, it reverts to the previous belief.
1.3 POMCP (Monte Carlo Tree Search)
POMCP (Partially Observable Monte Carlo Planning) is used for action selection. It works by:

Simulating different action sequences.
Estimating rewards for each action by averaging the outcomes of multiple simulations.
Selecting the best action based on expected total reward.
At the root level, POMCP prints detailed debug outputs, showing:

Immediate reward for taking an action.
Future value (expected reward from deeper simulations).
Total value (sum of immediate reward and future value).
Best action at the root based on simulations.
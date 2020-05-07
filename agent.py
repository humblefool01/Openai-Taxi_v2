import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.gamma = 1.0
        self.epsilon = 0.1
        self.decay = 0.99
        self.min_decay = 0.005
        
    def get_policy_probability(self, state):
        self.epsilon = max(self.min_decay, self.epsilon * self.decay)
        probs = np.ones(self.nA) * self.epsilon / self.nA
        probs[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)                
        return probs
    
    def select_action(self, state, env):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """        
        probs = self.get_policy_probability(state)
        return np.random.choice(np.arange(self.nA), p=probs)
         # return np.argmax(self.Q[state])
        
    def step(self, state, action, reward, next_state, done, env):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
#         old_value = self.Q[state][action]
#         next_max = np.max(self.Q[next_state])    
#         new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
#         self.Q[state][action] = new_value
        
        next_action = self.select_action(state, env)
        probs = self.get_policy_probability(state)
        
        # SARSA
        # self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]) 
        # SARSA_MAX
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        # Expected SARSA
        # self.Q[state][action] += self.alpha * (reward + self.gamma * np.sum(self.Q[next_state] * probs) - self.Q[state][action])
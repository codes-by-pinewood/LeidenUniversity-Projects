#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep#
        done indicates whether the final s in states is was a terminal state '''
        episode_length = len(states) -1  
        G=np.zeros(episode_length)
    
        for t in range(episode_length):

            m = min(n, episode_length-t)
            # DO we evaluate whether done ?
            if done:
                G[t] += sum([pow(self.gamma,i) * rewards[t+i] for i in range(m)])
                    
            else: 
                G[t] = sum([pow(self.gamma,i) * rewards[t+i] for i in range(m)]) + pow(self.gamma, m) * max(self.Q_sa[states[t+m]])

            self.Q_sa[states[t]][actions[t]] += self.learning_rate * (G[t] - self.Q_sa[states[t]][actions[t]])
    


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    s = env.reset()

    env.Q_sa = np.zeros((env.n_states, env.n_actions))
    #n_timesteps =  n_timesteps//4
    budget = n_timesteps

    while budget > 0:
        states_array, actions_array, rewards_array= [s], [], []
        for t in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp) # initialize with policy 
            s_next, r, done = env.step(a) # step 
            #append the attributes of step to arrays
            states_array.append(s_next)
            rewards_array.append(r)
            actions_array.append(a)
            s = s_next
            budget -=1
                
            if done:
                break 
            
            #add to plot variables
            if ((n_timesteps-budget)%eval_interval == 0):
                #print("NSTEP: {}".format(n_timesteps-budget))
                ans = pi.evaluate(eval_env)
                eval_returns.append(ans)
                eval_timesteps.append(n_timesteps - budget)
            
        
        pi.update(states_array,actions_array, rewards_array, done, n)


    return eval_returns, eval_timesteps 
        

def test():
    n_timesteps = 50001
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print(eval_returns)

    
    
if __name__ == '__main__':
    test()

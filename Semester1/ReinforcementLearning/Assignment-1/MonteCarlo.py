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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        episode_length = len(states) -1  
        #print(episode_length)
        G = np.zeros(episode_length+1)

        sum_rewards = 0
        for i in reversed(range(episode_length)):
            G[i] = rewards[i] + self.gamma*G[i+1]
            self.Q_sa[states[i]][actions[i]] = self.Q_sa[states[i]][actions[i]] + self.learning_rate * (G[i] - self.Q_sa[states[i]][actions[i]] )


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=100):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    states_array = []
    actions_array = []
    rewards_array = []

    budget  = n_timesteps 
    newbudget = budget
    
    #loop over budget
    while budget > 0: 
        states_array = []
        actions_array = []
        rewards_array = []
        s = env.reset()
        states_array.append(s)
        for t in range(min(budget, max_episode_length)):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            budget = budget - 1
            #append the attributes of step to arrays
            states_array.append(s_next)
            actions_array.append(a)
            rewards_array.append(r)

            if done: 
                break
            
            #add to plot variables
            if ((n_timesteps-budget)%eval_interval == 0):
                print("MC: {}".format(n_timesteps-budget))
                #print("montecarlo ntimesteps -budget  {}".format(n_timesteps - budget))
                eval_returns.append(pi.evaluate(eval_env))
                eval_timesteps.append((n_timesteps-budget))
         
        
        pi.update(states_array,actions_array, rewards_array)
     
       

       
            



    # TO DO: Write your Monte Carlo RL algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return eval_returns, eval_timesteps
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print(eval_returns, eval_timesteps)
    
            
if __name__ == '__main__':
    test()

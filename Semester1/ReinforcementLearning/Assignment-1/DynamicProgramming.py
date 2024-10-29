#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
    
    def full_argmax(x):
        return np.where(x == np.max(x))[0] 
    
    def _state_to_location(self,state):
        ''' bring a state index to an (x,y) location of the agent '''
        return np.array(np.unravel_index(state,self.shape))
                
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        a = QValueIterationAgent.full_argmax(self.Q_sa[s])
        # if there is more than one max action, then choose 1 at random
        if (len(a) > 1):
            a = np.random.choice(a)
        else:
            a = a[0]

        #total_reward += self.Q_sa[s][a]
        return a
    
    def plot_qsa(a):
        length = len(a)
        x = np.linspace(0,1,length)
        y = a
       
        # y = a
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y, color="lightblue", linewidth=3)
      
        
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''        
        val = np.sum(p_sas*(r_sas + self.gamma*np.max(self.Q_sa, axis=1)))
        return val
  


    
def calculate(env, state):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    max_return = 0
    for a in range(env.n_actions):
        p_sas,r_sas = env.model(state,a)
        max_return += p_sas[s]*(r_sas)
    
    return max_return
        
        
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    new_Q_sa = np.zeros((env.n_states, env.n_actions))
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)  
    old_Q_sa = np.zeros((env.n_states, env.n_actions))
    delta = 1
    itr = 0


    while (delta > threshold):
        delta = 0
        itr = itr+1
        #sweep through state-action pairs
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s,a)
                old_Q_sa = QIagent.Q_sa[s][a] 
                new_Q_sa[s][a] = QValueIterationAgent.update(QIagent, s, a, p_sas, r_sas)
                delta = max(delta, abs(old_Q_sa - new_Q_sa[s][a]))
                QIagent.Q_sa[s][a] = new_Q_sa[s][a]
                    
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.0001)
        
    

    return QIagent



def experiment():
    
        gamma = 1.0
        threshold = 0.001

        env = StochasticWindyGridworld(initialize_model=True)
        env.render()
        QIagent= Q_value_iteration(env,gamma,threshold)
        
    
            # view optimal policy
        done = False
        s = env.reset()
        #calculations for total reward and average number of steps:
        #total_reward = 0
        #no_of_steps = 0
       
        while not done:
        
            a = QIagent.select_action(s)
            #no_of_steps = no_of_steps + 1
            s_next, r, done = env.step(a)
            env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=1)
            s = s_next

           
        

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()

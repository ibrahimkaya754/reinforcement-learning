#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Modules
import time
import math
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
from pygame.locals import *
from swarm import *
from DDPG import *


# In[2]:


# Simulation Parameters
number_of_particles = 24
number_of_axes      = 2
delta_t             = 0.5
t_final             = 5000
screen_size         = [800,600]
initial_location    = [screen_size[0]/2,screen_size[1]/2]
initial_location    = [0,0]
list_min_distance   = []
list_ave_distance   = []
xtrg                = [initial_location[ii] + np.random.randint([500,100])[ii] for ii in range(number_of_axes)]
particles           = swarm(number_of_particles=number_of_particles, screensize=screen_size, target_location=xtrg,
                            display=True, CommRng=100, dim=number_of_axes)
rlagent             = [key for key in particles.member.keys() if particles.member[key]['role']=='rlagent'][0]
leader              = particles.leader
numberofneighbour   = 5
numberofleader      = 1
clock               = pygame.time.Clock()
numberofepochs      = 5
state               = []
newstate            = []
train               = True


# In[3]:


# Instance of DDPG is created.
print('----------------------------------------------------------------------------')
print('There will be %s states, %s for relative velocity, %s for relative position' % \
      (particles.dim*(numberofneighbour+numberofleader)*2,\
      particles.dim*(numberofneighbour+numberofleader),\
      particles.dim*(numberofneighbour+numberofleader)))
print('----------------------------------------------------------------------------')
### Some states are from the closest leader ###
print('%s of the states are gathered from the closest leader of the swarm' % (numberofleader*particles.dim*2))
print('----------------------------------------------------------------------------')
myagent           = Agent(actor_network  = {'nn'          :[300,200],
                                            'activation'  :'relu',
                                            'initializer' :glorot_normal,
                                            'optimizer'   :Adam(learning_rate=0.001)}, 
                          critic_network = {'nn'          :[200,300],
                                            'concat'      :[100,200,50],
                                            'activation'  :'relu',
                                            'initializer' :glorot_normal,
                                            'optimizer'   :Adam(learning_rate=0.002)},
                          loadsavedfile=False,
                          disablenoise=False,
                          lowerBound=-3,upperBound=3,
                          numberOfActions=number_of_axes,
                          numberOfStates=particles.dim*(numberofneighbour+numberofleader)*2,
                          buffer_capacity= 50000, batch_size= 256,
                          tau= 0.005, gamma= 0.1, annealing= 1000)


# In[4]:


#States are appended to the "states list"
def stateappend(state):
    state = []
    for relpos,relvel in zip(list(particles.member[rlagent]['relative_position'].values())[0:numberofneighbour],\
                             list(particles.member[rlagent]['relative_velocity'].values())[0:numberofneighbour]):
        for pos,vel in zip(relpos.values(),relvel.values()):
            state.append(pos)
            state.append(vel)

    for relpos,relvel in zip(list(particles.member[rlagent]['distance2leader'].values()),\
                             list(particles.member[rlagent]['velocity2leader'].values())):
        state.append(relpos)
        state.append(relvel)
    state = np.array(state)
    return state


# In[5]:


# Reward Function 
def rewardfunction(dist2leader,dist2closest,score,t):
    if dist2leader >= 500.0 or dist2closest <=2.0:
        reward = -10000
    else:
        if dist2closest > 2.0 and dist2closest < 10.0:
            reward = dist2closest**3 - dist2leader
        else:
            reward = 1000 - dist2closest**1.5 - dist2leader
    
    reward = reward / 10000
    score = score + reward
    t  = t + delta_t
    
    if score <= -100 or reward <= -1 or t >= t_final:
        done = True
    else:
        done = False
    
    return reward, score, done, t


# In[ ]:


# Main Function
for epoch in range(numberofepochs):
    xtrg        = [np.random.randint(screen_size)[ii] for ii in range(number_of_axes)]
    particles.__init__(number_of_particles=number_of_particles,screensize=screen_size,target_location=xtrg,
                       display=True,CommRng=100,summary=False)
    rlagent     = [key for key in particles.member.keys() if particles.member[key]['role']=='rlagent'][0]
    state       = stateappend(state)
    done        = False
    t, score    = 0 , 0 
    myagent.mtd = False
    myagent.msd = False

    print('-----------------------------')

    cum_sum_of_every_particles = 0
    for i_particles in list(particles.member.keys())[1:]:
        print(particles.member[i_particles].keys())
        for each_abs_distance_sorted in particles.member[i_particles]['abs_distance_sorted'][1:]:
            cum_sum_of_every_particles += each_abs_distance_sorted
        
    avg_of_every_particles = cum_sum_of_every_particles / len(list(particles.member.keys())[1:])


    print('-----------------------------')


    
    while not done:
        particles.rulebasedalgo()
        action = myagent.policy(state.reshape(1,myagent.numberOfStates))

        for dim in range(particles.dim):
            particles.member[rlagent]['deltavel'][str(dim)] = action[int(dim)]

        particles.update(keepGoing=not done)
        #distance = {'2leader'  : (lambda x: np.sqrt(x[0]**2+x[1]**2))\
        #                         (list(particles.member[rlagent]['distance2leader'].values())),
        #            '2closest' : particles.member[rlagent]['abs_distance_sorted'][1]}
        
        distance = {'2leader'  : (lambda x: np.sqrt(x[0]**2+x[1]**2))\
                                 (list(particles.member[rlagent]['distance2leader'].values())),


                    #### x ve y'nin birlikte değerlerinin alınması gerekmez mi?
                    
                    #articles.member[rlagent]['abs_distance_sorted'][1]
                    '2closest' : avg_of_every_particles}
        
        newstate = stateappend(newstate)
        
        reward, score, done, t = rewardfunction(distance['2leader'],distance['2closest'],score,t)
        myagent.observation    = (state,action,reward,newstate)
        myagent.record_buffer()
        
        state = newstate.copy()
        print('ep= %s, act0= %.3f, act1= %.3f, vel0= %.3f, vel1= %.3f, rwd= %0.2f, scr= %0.2f, mt= %0.2f, ms= %0.2f, d2l= %0.2f, d2c= %0.2f, noisevar= %.2f, t= %0.1f' %\
             (epoch,action[0],action[1],particles.member[rlagent]['velocity']['0'],particles.member[rlagent]['velocity']['1'],reward,score,myagent.maxtime,myagent.maxscore,distance['2leader'],distance['2closest'],myagent.noisevariance,t))
        
        if t%100 >= 0.0 and t%100 < delta_t:
            print('\ntarget location changes\n')
            particles.trgt_loc                 = {str(ii) : np.random.randint(screen_size)[ii] for ii in\
                                                            range(particles.dim)}
            particles.targetposition['target'] = particles.trgt_loc

    if done:
        if t >= myagent.maxtime:
            myagent.maxtime  = t
            myagent.mtd      = True
            print('saving models for mtd')
        if score >= myagent.maxscore:
            myagent.maxscore = score
            myagent.msd      =True
            print('saving models for msd')
        if train:
            myagent.save()
            myagent.learn()
    
        print('\n----- New Epoch ----- Epoch: %s\n' % (epoch+1))

    
    print('-----------------------------------------------------------------')


clock.tick(60)
time.sleep(15)
# In[ ]:





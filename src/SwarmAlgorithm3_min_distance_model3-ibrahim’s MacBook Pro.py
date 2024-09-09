'''
Author: ikaya
'''
#%%
# Import neccessary modules
import time
import math
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
from pygame.locals import *
from swarm import *
from QLearningClass import *

# Simulation Parameters
number_of_particles = 24
number_of_axes      = 2
delta_t             = 0.1
t_final             = 2000

def actions():
    act = np.ndarray(shape=(11,2))
    ctr = 0
    for ii in range(11):
       act[ii,0] = ii
       act[ii,1] = (ctr-5)/10
       ctr = ctr + 1
    return act

screen_size       = [1400,900]
initial_location  = [screen_size[0]/2,screen_size[1]/2]
list_min_distance = []
list_ave_distance = []
xtrg              = [initial_location[ii] + np.random.randint([1400,900])[ii] for ii in range(number_of_axes)]
particles         = swarm(number_of_particles=number_of_particles, screensize=screen_size, target_location=xtrg,
                          display=True, CommRng=100, dim=number_of_axes)
rlagent           = [key for key in particles.member.keys() if particles.member[key]['role']=='rlagent'][0]
leader            = particles.leader
numberofneighbour = 5
numberofleader    = 1
clock             = pygame.time.Clock()
numberofepochs    = 10000

### The multiplayer 2 below is for 'position' and 'velocity' ###
print('----------------------------------------------------------------------------')
print('There will be %s states, %s for relative velocity, %s for relative position' % \
      (particles.dim*(numberofneighbour+numberofleader)*2,\
      particles.dim*(numberofneighbour+numberofleader),\
      particles.dim*(numberofneighbour+numberofleader)))
print('----------------------------------------------------------------------------')
### Some states are from the closest leader ###
print('%s of the states are gathered from the closest leader of the swarm' % (numberofleader*particles.dim*2))
print('----------------------------------------------------------------------------')
action            = actions()  
myagent           = agent(numberofstate=particles.dim*(numberofneighbour+numberofleader)*2,numberofmodels=5,
                          numberofaction=len(action),load_saved_model=False,dim=number_of_axes,list_nn=[100,250,50], gamma=0.8, tau=0.01, buffer=100000)
################################################
### States are appended to the "states list" ###
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
###############################################
### Reward Function ###
def reward(dist2leader,dist2closest,score,time):
    if dist2leader >= 500.0 or dist2closest <=2.0:
        reward = -10000
    else:
        if dist2closest > 2.0 and dist2closest < 10.0:
            reward = dist2closest**3 - dist2leader
        else:
            reward = 1000 - dist2closest**1.5 - dist2leader
    score = score + reward
    time  = time + delta_t
    if score <= -1000000 or reward <= -10000 or time >= t_final:
        done = True
    else:
        done = False
    return reward, score, done, time
################################################
#%%
for epoch in range(numberofepochs):
    xtrg          = [np.random.randint(screen_size)[ii] for ii in range(number_of_axes)]
    particles.__init__(number_of_particles=number_of_particles,screensize=screen_size,target_location=xtrg,
                       display=True,CommRng=100,summary=False)
    rlagent       = [key for key in particles.member.keys() if particles.member[key]['role']=='rlagent'][0]
    myagent.state = stateappend(myagent.state)
    myagent.done  = False
    t, score      = 0 , 0 
    while not myagent.done:
        particles.rulebasedalgo()
        qval    = myagent.model['model1']['model_network'].predict(myagent.state.reshape(1,myagent.numberofstate))
        actionn = []
        for dim in range(particles.dim):
            randomnumber = np.random.random()
            if  randomnumber < myagent.epsilon:
                actionn.append(np.random.randint(0,myagent.numberofaction))
            else:
                actionn.append(int(np.argmax(qval[dim])))
            particles.member[rlagent]['deltavel'][str(dim)] = action[int(actionn[dim])][1]
        particles.update(keepGoing=not myagent.done)
        distance         = {'2leader'  : (lambda x: np.sqrt(x[0]**2+x[1]**2))\
                                         (list(particles.member[rlagent]['distance2leader'].values())),
                            '2closest' : particles.member[rlagent]['abs_distance_sorted'][1]}
        myagent.newstate = stateappend(myagent.newstate)
        myagent.reward, score, myagent.done, t = reward(distance['2leader'],distance['2closest'],score,t)
        myagent.replay_list(actionn)
        myagent.state    = myagent.newstate
        myagent.epsilon  = 0.05 if myagent.epsilon<0.05 else myagent.epsilon - 1 / myagent.buffer
        print('epoch= %s, act1= %0.2f, act2= %0.2f ,reward= %0.2f, score= %0.2f, maxtime= %0.2f, maxscore= %0.2f, dist2leader= %0.2f, dist2closest= %0.2f, eps= %.2f, time= %0.1f' %\
             (epoch,action[int(actionn[0])][1],action[int(actionn[1])][1],myagent.reward,score,myagent.maxtime,myagent.maxscore,distance['2leader'],distance['2closest'],myagent.epsilon,t))
        
        if t%100 >= 0.0 and t%100 < delta_t:
            print('\ntarget location changes\n')
            particles.trgt_loc                 = {str(ii) : np.random.randint(screen_size)[ii] for ii in\
                                                            range(particles.dim)}
            particles.targetposition['target'] = particles.trgt_loc

    if myagent.done:
        if t >= myagent.maxtime:
            myagent.maxtime  = t
            myagent.mtd      = True
        else:
            myagent.mtd      = False
        if score >= myagent.maxscore:
            myagent.maxscore = score
            myagent.msd      = True
        else:
            myagent.msd      = False
        
        
        myagent.train_model(epoch=epoch)
        myagent.save(time=t, target_time=150, score=score, target_score=100000)
   
        replay = myagent.save_replay
        print('buffer size= %s' % (len(myagent.replay)))
        print('\n----- New Epoch ----- Epoch: %s\n' % (epoch+1))
        #%%                
print('\n------------------------------------')
for key in particles.member.keys():
    print('Particle id       : %s' % (key))
    print('Particle role     : %s' % (particles.member[key]['role']))
    print('Particle color    : ', particles.color[particles.member[key]['role']])
    print('Particle target   : %s' % (particles.member[key]['target']))
    print('particle velocity : %s' % (particles.member[key]['velocity']))
    print('particle position : %s' % (particles.member[key]['position']))
    print('target position   : %s' % (particles.targetposition[particles.member[key]['target']]))
    print('weigths           : %s' % (particles.wght[particles.member[key]['role']]))
    print('particles in rng  : %s' % (particles.member[key]['PrtclsInRng']))
    print('------------------------------------')

# %%

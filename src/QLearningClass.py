"""
Created on Wed Nov  8 15:13:04 2017
@author: ikaya
"""
import sys
import warnings
import gym
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import random
warnings.filterwarnings("ignore")


print('Done!')

class neuralnet():
    def __init__(self, numberofstate, numberofaction, 
                 activation_func, trainable_layer, initializer,
                 list_nn, load_saved_model, numberofmodels, dim):
        
        self.activation_func  = activation_func
        self.trainable_layer  = trainable_layer
        self.init             = initializer
        self.opt              = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.regularization   = 0.0
        self.description      = ''
        self.numberofstate    = numberofstate
        self.numberofaction   = numberofaction
        self.list_nn          = list_nn
        self.load_saved_model = load_saved_model
        self.total_layer_no   = len(self.list_nn)+1
        self.numberofmodels   = numberofmodels
        self.loss             = 'mse'
        self.model            = {}
        self.input            = Input(shape=(self.numberofstate,), name='states')
        self.dim              = dim

        print('\nCreating RL Agents\n')
        LOut = {}
        for ii in range(self.numberofmodels):
            model_name = 'model'+str(ii+1)
            model_path = os.getcwd()+"/" + model_name + '.keras'
            L1 = Dense(self.list_nn[0], activation=self.activation_func,
                       kernel_initializer=self.init, trainable = self.trainable_layer)(self.input)

            for ii in range(1,len(self.list_nn)):
                L1 = Dense(self.list_nn[ii], activation=self.activation_func, trainable = self.trainable_layer,
                           kernel_initializer=self.init)(L1)    

            for dimension in range(self.dim):
                LOut['action'+str(dimension)]  = Dense(self.numberofaction, activation='linear', name='action'+str(dimension),
                            kernel_initializer=self.init)(L1)
            
            model = Model(inputs=self.input, outputs=[LOut['action'+str(dimension)] for dimension in range(self.dim)])
            tf.keras.utils.plot_model(model,to_file=model_name+'.png', show_layer_names=True,show_shapes=True)
            print('\n%s with %s params created' % (model_name,model.count_params()))

            optimizer = tf.keras.optimizers.Adam()

            model.compile(optimizer=optimizer, loss=self.loss, metrics=['mse'] * self.dim)

            self.model[model_name] = { 'model_name'    : model_name,
                                       'model_path'    : model_path,
                                       'model_network' : model,
                                       'numberofparams': model.count_params()}
                                       
            self.model['model1']['best'] = { 'model_path'    : {'maxscore' : os.getcwd()+"/" + 'best_model_msd' + '.keras',
                                                               'maxtime'  : os.getcwd()+"/" + 'best_model_mtd' + '.keras'},
                                            'model_network' : {'maxscore' : '','maxtime'  : ''},
                                            'mtd'           : False,
                                            'msd'           : False,
                                            'maxtime'       : 0,
                                            'maxscore'      : 0 }
            if self.load_saved_model:
                if not os.path.exists(self.model['model1']['model_path']):
                    print('There is no model saved to the related directory!')
                else:
                    self.model[model_name]['model_network'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxtime']  = load_model(self.model['model1']['model_path'])
                                            
                    if os.path.exists(self.model['model1']['best']['model_path']['maxscore']):
                        self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['best']['model_path']['maxscore'])
                    if os.path.exists(self.model['model1']['best']['model_path']['maxtime']):
                        self.model['model1']['best']['model_network']['maxtime']  = load_model(self.model['model1']['best']['model_path']['maxscore'])
        print('\n-----------------------')
        self.listOfmodels = [key for key in self.model.keys()]
    def __describe__(self):
        return self.description
     
    def summary(self):
        for key in self.model.keys():
            self.model[key]['model_network'].summary()
            print('\nModel Name is: ',self.model[key]['model_name'])
            print('\nModel Path is: ',self.model[key]['model_path'])
            print('\nActivation Function is: ',self.activation_func)
            print('\n*******************************************************************************')
        if self.description != '':
            print('\nModel Description: '+self.__describe__())

class agent(neuralnet):
    def __init__(self, numberofstate, numberofaction, dim, activation_func='elu', trainable_layer= True, 
                 initializer= 'he_normal', list_nn= [250,150], 
                 load_saved_model= False, location='./', buffer= 50000, annealing= 5000, 
                 batchSize= 100, gamma= 0.95, tau= 0.001, numberofmodels= 2):
        
        super().__init__(numberofstate=numberofstate, numberofaction=numberofaction, activation_func=activation_func,
                         trainable_layer=trainable_layer, initializer=initializer,list_nn=list_nn, 
                         load_saved_model=load_saved_model, numberofmodels=numberofmodels, dim=dim)
        
        self.epsilon                  = 1.0
        self.location                 = location
        self.gamma                    = gamma
        self.batchSize                = batchSize
        self.buffer                   = buffer
        self.annealing                = annealing
        self.replay                   = []
        self.sayac                    = 0
        self.tau                      = tau
        self.state                    = []
        self.reward                   = None
        self.newstate                 = None
        self.done                     = False
        self.maxtime                  = 0
        self.maxscore                 = 0
        self.mtd                      = False
        self.msd                      = False
        
    def replay_list(self,actionn):
        if len(self.replay) < self.buffer: #if buffer not filled, add to it
            self.replay.append((self.state, actionn, self.reward, self.newstate, self.done))
            #print("buffer_size = ",len(self.replay))
        else: #if buffer full, overwrite old values
            if (self.sayac < (self.buffer-1)):
                self.sayac = self.sayac + 1
            else:
                self.sayac = 0
            self.replay[self.sayac] = (self.state, actionn, self.reward, self.newstate, self.done)
            #print("sayac = ",self.sayac)

    def soft_update(self, main_model, target_model):
        main_weights = main_model.get_weights()
        target_weights = target_model.get_weights()
        new_weights = []
        for main_weight, target_weight in zip(main_weights, target_weights):
            new_weight = self.tau * main_weight + (1 - self.tau) * target_weight
            new_weights.append(new_weight)
        target_model.set_weights(new_weights)

    def remember(self, main_model, target_model):
        model = self.model[main_model]
        target_model = self.model[target_model]
        minibatch = random.sample(self.replay, self.batchSize)

        # Initialize arrays for batch processing
        states_old = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        states_new = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])

        # Predict Q-values for old and new states
        Qval_old = model['model_network'].predict(states_old)
        Qval_new = model['model_network'].predict(states_new)
        Qval_trgt = target_model['model_network'].predict(states_new)

        # Initialize y_train
        y_train = [Qval_old[dim].copy() for dim in range(self.dim)]

        for i in range(self.batchSize):
            for dim in range(self.dim):
                action = np.argmax(Qval_new[dim][i])
                maxQ = Qval_trgt[dim][i][action]
                if not dones[i]:
                    update = rewards[i] + self.gamma * maxQ
                else:
                    update = rewards[i]
                y_train[dim][i][actions[i][dim]] = update

        # Train the model
        model['model_network'].fit(states_old, y_train, batch_size=self.batchSize, epochs=1, verbose=1)
        
        # Perform soft update on the target network
        self.soft_update(model['model_network'], target_model['model_network'])

        return model

    def train_model(self, epoch):
        '''
        This function is used to train the model. It is called after the agent is done with the episode.

        '''
        print('\n%s and %s are main and target models, respectively' % ('model1','model2'))
        self.remember('model1','model2')
        print('Training is done')
        if epoch % 10 == 0:
            counter1 = 1
            counter2 = counter1 + 1
            for _ in range(self.numberofmodels-1):
                if counter2 >= self.numberofmodels:
                    counter2 = 0
                print('%s and %s are main and tardet models, respectively' % (self.listOfmodels[counter1],self.listOfmodels[counter2]))
                self.remember(self.listOfmodels[counter1],self.listOfmodels[counter2])
                counter1 = counter1 + 1
                counter2 = counter2 + 1    
            print('Training is done for all models')      
     

        if len(self.replay) >= self.annealing:        
            print('Training is done')
        else:
            print('Training will begin after %s data gathered' % (self.annealing))    

    def save_replay(self):
        return self.replay
        
    def save(self, time, target_time, score, target_score):
        self.model['model1']['model_network'].save(self.model['model1']['model_path'])

        if self.mtd:
            self.model['model1']['best']['mtd']                      = self.mtd
            self.model['model1']['best']['maxtime']                  = self.maxtime
            self.model['model1']['best']['model_network']['maxtime'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxtime'].save(self.model['model1']['best']['model_path']['maxtime'])

        if self.msd:
            self.model['model1']['best']['msd']                       = self.msd
            self.model['model1']['best']['maxscore']                  = self.maxscore
            self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxscore'].save(self.model['model1']['best']['model_path']['maxscore'])

        

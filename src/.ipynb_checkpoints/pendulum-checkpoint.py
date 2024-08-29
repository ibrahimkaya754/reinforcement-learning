import os
import subprocess

print("DISPLAY:", os.environ.get("DISPLAY"))

from DDPG import *
import warnings

warnings.simplefilter("ignore")
# Learning rate for actor-critic models
total_episodes   = 100
ep_reward_list   = [] # To store reward history of each episode
avg_reward_list  = [] # To store average reward history of last few episodes
best_reward      = -10e8

env = gym.make("Pendulum-v1", render_mode='human')

#  Get the number od states and actions
no_of_states = env.observation_space.shape[0]
no_of_actions = env.action_space.shape[0]
a_bound = env.action_space.high

myagent = Agent(environment=env,loadsavedfile=False,disablenoise=True,numberOfStates=no_of_states,
                numberOfActions=no_of_actions, upperBound=a_bound,lowerBound=-a_bound, annealing=250)

for eps in range(total_episodes):
    state, _        = env.reset()
    episodic_reward = 0

    while True:
        env.render()
        state  = np.array(state)
        action = myagent.policy(state.reshape(1, no_of_states))
        step_result = env.step(action)

        if len(step_result) == 4:
            nextstate, reward, done, info = step_result
        elif len(step_result) == 5:
            nextstate, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            raise ValueError("Unexpected number of values returned from env.step()")    

        myagent.observation = (state,action,reward,nextstate)
        myagent.record_buffer()
        episodic_reward = episodic_reward + reward
        myagent.learn()
        # End this episode when `done` is True
        if done:
            print('noise variance is %.8f' % (myagent.noisevariance))
            print('noise is %.8f' % (myagent.noise))
            break

        state = nextstate

    ep_reward_list.append(episodic_reward)
    # Mean of last 10 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(eps, avg_reward))
    avg_reward_list.append(avg_reward)
    if avg_reward_list[-1]>best_reward:
        best_reward = avg_reward_list[-1]
        print('saving models')
        #myagent.save()
    print('-----------------------------------------------------------------')
# Plotting graph Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
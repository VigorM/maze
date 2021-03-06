import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym
import gym_maze

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

# Seed는 random number를 만들어 줄 때, 완벽히 랜덤은 아님
# Seed를 똑같이 설정하면 똑같은 난수가 생성, 지금 코드가 돌아가고 있는 시간을 받아서 Seed로 쓴다면 완전한 난수 가능

def select_action(state,plot=False, steps=0):
    global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END)* \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # eps_threshold 값에 100을 곱한 확률로 Agent가 랜덤으로 action을 취함
    eps_threshold = 0.05
    steps_done += 1
    if sample > eps_threshold and steps_done > 1000:
        with torch.no_grad():
            state_preprocess = torch.unsqueeze(torch.unsqueeze(state.to('cuda'),0),0)
            # a = policy_net(torch.unsqueeze(state.to('cuda'),0)).view(1,1)

            # return policy_net(torch.unsqueeze(state.to('cuda'),0)).max(1)[1].view(1, 1)
            return policy_net(state_preprocess,plot,steps).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """e
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    states = tuple((map(lambda s: torch.unsqueeze(torch.unsqueeze(s,0),0), batch.state)))
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = tuple((map(lambda s: torch.unsqueeze(torch.unsqueeze(s,0),0), non_final_next_states)))
    non_final_next_states = torch.cat(non_final_next_states).to('cuda')
    

    state_batch = torch.cat(states).to('cuda')

    # state_batch = torch.tensor(batch.state, device='cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    # obs = env.maze_view.map
    state = np.array(obs)
    # state = state.transpose((2, 0, 1))
    state = torch.from_numpy(obs)
    # print(state)
    return state

def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        env.maze_change()
        obs = env.reset()
        plotnow = False
        # state = env.maze_view.map
        state = get_state(obs)
        total_reward = 0.0
        for t in count():

            # if episode % 1 == 0:
            #     render = True
            #     plotnow = True
            if episode % 10 == 0:
                render = True
                if t % 500 == 0:
                    plotnow = True
                else:
                    plotnow = False
            else:
                render = False
                plotnow = False 
            if render:
                env.render()

            action = select_action(state, plotnow, steps_done)

            robot_pos_bef = env.maze_view.robot
            obs, reward, done, info = env.step(action)
            robot_pos_aft = env.maze_view.robot
            if np.all(robot_pos_aft == robot_pos_bef):
               reward += -0.2/(env.maze_size[0]*env.maze_size[0])
            
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done % 500 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))

            if steps_done > INITIAL_MEMORY:
                plotnow = True
                optimize_model()
                # Target network update
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
                break
        # if episode % 20 == 0:
        #         print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward)) # t는 episode의 time step
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    # env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        # env.maze_change()
        obs = env.reset()
        # my_state = env.maze_view.map
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            state_preprocess = torch.unsqueeze(torch.unsqueeze(state.to('cuda'),0),0)
            action = policy(state_preprocess).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 4 # 10000에서 수정
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10
    MEMORY_SIZE = 10 * INITIAL_MEMORY # 100 * INITAIL_MEMORY

    # create networks
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0

    # create environment
    env = gym.make('maze-random-10x10-v0')
    # env = make_env(env)
    
    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, 40, False)
    torch.save(policy_net, "dqn_maze_model_v2.pt")
    policy_net = torch.load("dqn_maze_model_v2.pt")
    test(env, 1, policy_net, render=False)
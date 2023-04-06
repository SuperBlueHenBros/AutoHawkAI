import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
from os import listdir


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(4,32,8,4)
      #self.relu1 = nn.ReLU()
      self.conv2 = nn.Conv2d(32,64,4,2)
      #self.relu2 = nn.ReLU()
      self.conv3 = nn.Conv2d(64,128,3,1)
      #self.relu3 = nn.ReLU()
      #self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(6272, 512)
      #self.relu4 = nn.ReLU()
      self.fc2 = nn.Linear(512,4)


    # x represents our data
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.conv3(x)
      x = F.relu(x)

      # Flatten x with start_dim=1
      x = torch.flatten(x)
     
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      # Output
      output = x
      return output
myNetwork = Net()

#Putting the model on the gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myNetwork.to(device)

#Parameters
iter = 0
num_iter = 10
num_actions = 4 #Subject to change, depends on game
replay_size = 1000 
minibatch_size = 32
memory_replay = []
epsilon = 0.4
gamma = 0.99
action = 0


#Actual iteration of the AI 

#Frame stacking by pulling image and advancing state 4 times
reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
#pull image data and preprocess image
image_1 = get_image
image_1 = prep_image    
reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
image_2 = get_image
image_2 = prep_image
reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
image_3 = get_image
image_3 = prep_image
reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
image_4 = get_image
image_4 = prep_image

state = torch.stack((image_1,image_2,image_3,image_4),0)




while iter < num_iter:

    output = myNetwork(state)

    #Might be able to put action on the gpu as well
    if torch.cuda.is_available():
        action = action.to(device)
    
    #Determining whether the action is random or ideal
    if np.random.random() < epsilon:
        action = argmax(output)
    else:
        action = np.random.randint(num_actions)
    
    
    
    #Frame stacking by pulling image and advancing state 4 times
    reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
    #pull image data and preprocess image
    image_1 = get_image
    image_1 = prep_image    
    reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
    image_2 = get_image
    image_2 = prep_image
    reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
    image_3 = get_image
    image_3 = prep_image
    reward = get_next_state(action)#doesn't need to get reward each time, the last time is the only one that matters
    image_4 = get_image
    image_4 = prep_image
    
    state_2 = torch.stack((image_1,image_2,image_3,image_4),0)

    memory_replay.append((state,action,reward,state_2))

    if len(memory_replay) > replay_size:
        memory_replay.pop()
    
    minibatch = random.sample(memory_replay, min(len(memory_replay)), replay_size)

    #Creating the separate batches
    state_batch = torch.cat(tuple(d[0] for d in minibatch))
    action_batch = torch.cat(tuple(d[1] for d in minibatch))
    reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

    if torch.cuda.is_available():
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        state_2_batch = state_1_batch.to(device)


    #Q values are obtained from the neural network
    q_values = myNetwork(state_batch).gather(1, action_batch.long().unsqueeze(1))
    next_q_values = myNetwork(state_2_batch).max(1)[0].detach()
    expected_q_values = reward_batch + gamma * next_q_values

    #Updates and others
    loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    state = state_2

    iter += 1
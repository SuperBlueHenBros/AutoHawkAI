from turtle import speed
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
from os import listdir
import middletier
import logging
import matplotlib.pyplot
import os.path
import mss 

def screenshot():
    with mss.mss() as sct:
        monitor = {"top": 53, "left": 2, "width": 585, "height": 450}
        #Top = how far from top
        #Left = how far from left
        #Width & height = widghtxheight
        #Can lock game screen and add a trim to specific size here
        img = sct.grab(monitor)
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (84, 84))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    return img_array


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(4,32,8,4)
      self.bn1 = nn.BatchNorm2d(32)
      self.conv2 = nn.Conv2d(32,64,4,2)
      self.bn2 = nn.BatchNorm2d(64)
      self.conv3 = nn.Conv2d(64,128,3,1)
      self.bn3 = nn.BatchNorm2d(128)
      self.fc1 = nn.Linear(6272, 512)
      self.fc2 = nn.Linear(512,6)

    # x represents our data
    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = self.bn2(x)
      x = F.relu(x)
      x = self.conv3(x)
      x = self.bn3(x)  
      x = F.relu(x)

      x = x.view(x.size(0), -1) #This line accounts for the batch size the x.size(0) is the batch size

      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      # Output
      output = x
      return output
    
myNetwork = Net()

loss = nn.MSELoss()
optimizer = torch.optim.Adam(myNetwork.parameters(), lr=1e-5)

#Putting the model on the gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myNetwork.to(device)

#Parameters
iter = 0
num_iter = 100000000
num_actions = 6 #Subject to change, depends on game
replay_size = 1000 
minibatch_size = 32
memory_replay = []
epsilon = 0.2
gamma = 0.99
action = 0
highest_x = 0
jumpCT = 0
passed_time = 0

# setup some basic config info
config_info = middletier.config.check() # data pulled from config.ini
game_path = config_info['directories']['data'] + "/NES/Mariobros.json"     # path to our game info (addreses and such)
rom_path = config_info['directories']['data'] + "/NES/roms/Super Mario Bros. (JU) (PRG0) [!].nes" # path to our ROM

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# save the main core of the middletier, this is how we'll control most of the program
client = middletier.Core(game_path, config_info, rom_path=rom_path)

# start up the emulator, specify the duration that we want to wait before continuing in seconds
client.spawn_emulator(startup_delay=10, load_state=True)

# initialize a local map of all addresses to read, and their current state
memory_state = {}
for addr in client.game.addresses:
    memory_state[addr] = 'None'

memory_map = {}
for map in client.game.mapping:
    memory_map[map['name']] = int(map['address'], base=16)

death = memory_state[memory_map['Players State']] == 11

#Actual iteration of the AI 
def next_frame(action_index, frames: int = 1): #This is for getting the next frame in the game. We may have the game do the same action for multiple frames.
    global jumpCT
    images = []
    capture = False
    counter = 0
    # press (0) and then release (1) the given button
    for frame in range(frames*2):
        if action == 0:
            client.send_input('P1 Down')
        elif action == 1:
            client.send_input('P1 Left')
        elif action == 2:
            client.send_input('P1 A', state=True)
            jumpCT = 0
        elif action == 3:
            client.send_input('P1 Right')
        elif action == 4:
            client.send_input('P1 B')

        if jumpCT >= 28:
            client.send_input('P1 A', state=False)
            jumpCT = 0

        jumpCT += 1
        
        client.conn.advance_frame()
        
        if frame % 2:
            images.append(screenshot()) # might need to turn into torch tensor
            

    # TODO: JUST TAKE WHAT WE NEED
    # loop through each address, read the state of the memory
    for addr in client.game.addresses:
        read_val = client.conn.read_byte(addr)
        try:
            memory_state[addr] = int(read_val.decode('ascii')[1:])
        except AttributeError:
            memory_state[addr] = 0
    
    speed = (memory_state[memory_map['PlayerHorizontal']]*256) + memory_state[memory_map['PlayerX']] #+ (memory_state[memory_map['CurrentScreen']] * 256) #+ memory_state[memory_map['ScreenEdge']]
    return speed, images

#Frame stacking by pulling image and advancing state 4 times
reward, screenshots = next_frame(action, frames=4) #doesn't need to get reward each time, the last time is the only one that matters
state = torch.Tensor(screenshots)


client.conn.save_state()
not_moving = 0
pit = memory_state[memory_map['VerticalScreenPos']]
while iter < num_iter:
    passed_time += 0.05
    if (memory_state[memory_map['Players State']] == 11) | (memory_state[memory_map['Players State']] == 6) | (not_moving >= 100) | (pit >= 4): # death to enemy or game over, does not include falling off map
        jumpCT = 0
        highest_x = 0
        prev_reward = 0
        reward = 0
        passed_time = 0
        not_moving = 0
        client.conn.load_state()

    state = state.unsqueeze(0)
    
    output = myNetwork(state)

    state = state.squeeze(0)

    #Might be able to put action on the gpu as well
    if torch.cuda.is_available():
        action = action.to(device)
    
    #Determining whether the action is random or ideal
    if np.random.random() > epsilon:
        action = argmax(output.detach().numpy())
    else:
        action = np.random.randint(num_actions)
    
    
    #Frame stacking by pulling image and advancing state 4 times
    reward, screenshots = next_frame(action, frames=4)
    prev_reward = reward
    if reward < 0:
        reward = 0
    if reward > highest_x:
        not_moving = 0
        reward = reward - highest_x
        highest_x = prev_reward
    elif highest_x == reward:
        not_moving += 1
        reward = -passed_time
    elif reward < highest_x:
        not_moving += 1
        reward = reward - highest_x - passed_time

    state_2 = torch.Tensor(screenshots)

    memory_replay.append((state,action,reward,state_2))

    if len(memory_replay) > replay_size:
        memory_replay.pop()

    
    if len(memory_replay) >= minibatch_size:
        minibatch = random.sample(memory_replay, minibatch_size)
        # Creating the separate batches
        state_batch = torch.stack((tuple(d[0] for d in minibatch)),0)
        action_batch = torch.tensor(tuple(d[1] for d in minibatch))
        reward_batch = torch.tensor(tuple(d[2] for d in minibatch))
        state_2_batch = torch.stack((tuple(d[3] for d in minibatch)),0)
        if torch.cuda.is_available():
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            state_2_batch = state_2_batch.to(device)

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

from turtle import speed
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
from os import listdir
# import numpy as np
import middletier
import logging
import matplotlib.pyplot
import os.path
import mss 

def screenshot():
    with mss.mss() as sct:
        monitor = {"top": 30, "left": 600, "width": 1000, "height": 700}
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
      #self.relu1 = nn.ReLU()
      self.bn1 = nn.BatchNorm2d(32)
      self.conv2 = nn.Conv2d(32,64,4,2)
      #self.relu2 = nn.ReLU()
      self.bn2 = nn.BatchNorm2d(64)
      self.conv3 = nn.Conv2d(64,128,3,1)
      #self.relu3 = nn.ReLU()
      self.bn3 = nn.BatchNorm2d(128)
      #self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(6272, 512)
      #self.relu4 = nn.ReLU()
      self.fc2 = nn.Linear(512,6)


    # x represents our data
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.conv3(x)
      x = F.relu(x)

      # Flatten x with start_dim=1
      #x = torch.flatten(x)
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

# setup some basic config info
config_info = middletier.config.check() # data pulled from config.ini
# print(f"config_info {config_info['directories']['data']}")
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

# client.send_input('P1 A', state=True)
#client.send_input('P1 Right', state=True)

#Actual iteration of the AI 
def next_frame(action_index, frames: int = 1): #This is for getting the next frame in the game. We may have the game do the same action for multiple frames.
    global jumpCT
    images = []
    capture = False
    counter = 0
    # press (0) and then release (1) the given button
    for frame in range(frames*2):
        #if action == 0: 
        #    client.send_input('P1 Up')
        if action == 0:
            client.send_input('P1 Down')
        elif action == 1:
            client.send_input('P1 Left')
        #elif action == 2:
        #    client.send_input('P1 Right')
        elif action == 2:
            client.send_input('P1 A', state=True)
            #for i in range(10):
            #client.send_input('P1 A', state=True)
                #client.conn.advance_frame()
            #client.send_input('P1 A', state=False)
            #client.send_input('P1 Right')
        elif action == 3:
            #client.send_input('P1 A')
            client.send_input('P1 Right', state=True)
            client.send_input('P1 Right', state=False)
        elif action == 4:
            client.send_input('P1 B')

        if jumpCT >= 16:
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
   # memory_state[memory_map['PlayerYPos']]-150 + memory_state[memory_map['PlayerXPos']]
    return speed, images

#Frame stacking by pulling image and advancing state 4 times
reward, screenshots = next_frame(action, frames=4) #doesn't need to get reward each time, the last time is the only one that matters


# TODO: don't do this
#reward -= 2

# [[image1],[image2],[image3],[image4]]

# state = torch.stack((screenshots[0], screenshots[1], screenshots[2], screenshots[3]),0)
state = torch.Tensor(screenshots)

#multiplier = 1
#flag = 0

# print(f"state: {state.shape}")
client.conn.save_state()


while iter < num_iter:
    if (memory_state[memory_map['Players State']] == 11) | (memory_state[memory_map['Players State']] == 6): # death to enemy or game over, does not include falling off map
        highest_x = 0
        prev_reward = 0
        reward = 0
        client.conn.load_state()
    
    prev_reward = reward

    state = state.unsqueeze(0)
    
    output = myNetwork(state)

    state = state.squeeze(0)

    #Might be able to put action on the gpu as well
    if torch.cuda.is_available():
        action = action.to(device)
    
    #Determining whether the action is random or ideal
    #if(iter < 100):
    #    action = 2
    if np.random.random() > epsilon:
        action = argmax(output.detach().numpy())
    else:
        action = np.random.randint(num_actions)
    
    
    
    #Frame stacking by pulling image and advancing state 4 times
    reward, screenshots = next_frame(action, frames=4) #doesn't need to get reward each time, the last time is the only one that matters
    if reward < 0:
        reward = 0
    if reward > highest_x:
        highest_x = reward
    elif highest_x == reward:
        reward = 0
    elif reward < highest_x:
        reward = reward - highest_x



    #highest_x = reward
    #if reward > prev_reward: #20 now, 15 prev
    #    reward = reward - prev_reward
    #    prev_reward = reward
    #elif reward < prev_reward:


    #reward = reward - prev_reward
    # TODO: don't do this
    #reward -= prev_reward
    #print("X Position:")
    print(reward)
    #print("Players State")
    #print(memory_state[memory_map['Players State']])
    #print("Highest X Value:")
    #print(highest_x)

    # state_2 = torch.stack((screenshots[0], screenshots[1], screenshots[2], screenshots[3]),0)
    state_2 = torch.Tensor(screenshots)
    # print(f"state_2: {state_2.shape}")

    memory_replay.append((state,action,reward,state_2))

    # print("after memory_replay.append()")

    if len(memory_replay) > replay_size:
        memory_replay.pop()

    # print("after memory_replay.pop()")
    
    if len(memory_replay) >= minibatch_size:
        # print("before minibatch")
        minibatch = random.sample(memory_replay, minibatch_size)
        # print("after minibatch")
        #Creating the separate batches
        # print("before state_batch")
        state_batch = torch.stack((tuple(d[0] for d in minibatch)),0)
        action_batch = torch.tensor(tuple(d[1] for d in minibatch))
        reward_batch = torch.tensor(tuple(d[2] for d in minibatch))
        state_2_batch = torch.stack((tuple(d[3] for d in minibatch)),0)
        # print("after state_1_batch")
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
# print("while loop done")

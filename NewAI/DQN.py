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


#Potential code for getting how far to the right the character goes
def image_displacement(image1, image2):
    image1 = image1.type(torch.uint8)
    image2 = image2.type(torch.uint8)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1.numpy(), None)
    kp2, desc2 = sift.detectAndCompute(image2.numpy(), None)

    # Match keypoints using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Filter good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return z
    # Estimate transformation using RANSAC
    src_pts = torch.FloatTensor([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = torch.FloatTensor([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts.numpy(), dst_pts.numpy(), cv2.RANSAC, 5.0)

    
    

    if M is not None:
        # Compute displacement in z direction
        z_disp = M[2, 0] / 8
    else:
        z_disp = 0

    return torch.tensor([z_disp], dtype=torch.float)

def screenshot():
    with mss.mss() as sct:
        monitor = {"top": 240, "left": 725, "width": 580, "height": 255}
        #Top = how far from top
        #Left = how far from left
        #Width & height = widghtxheight
        #Can lock game screen and add a trim to specific size here
        img = sct.grab(monitor)
        

        img_array = np.array(img)
        img_array = cv2.resize(img_array, (84, 84))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    return img_array

def calc_epsilon(episode):
    decay = (init_epsilon - end_epsilon) / epsilon_decay
    new_epsilon = max(end_epsilon, init_epsilon - (decay * episode))
    return new_epsilon

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
      self.fc2 = nn.Linear(512,5)


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

#Separate target network
TargetNetwork = Net()

loss = nn.MSELoss()
optimizer = torch.optim.Adam(myNetwork.parameters(), lr=0.0001)

#Putting the model on the gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myNetwork.to(device)

#putting target network on gpu
TargetNetwork.to(device)

#Loading the model if it has had any training done
if os.path.isfile('saved_weights.pth'):
    print("weights loaded")
    myNetwork.load_state_dict(torch.load('saved_weights.pth'))

#Load target as normal in the beginning
if os.path.isfile('saved_weights.pth'):
    print("target weights loaded")
    TargetNetwork.load_state_dict(myNetwork.state_dict())

#Attempt at implementing game over remedy
#if os.path.isfile('GameFin.pt'):
#    print("Finish game image loaded")
#    Finish = torch.load('GameFin.pt')


#Parameters
iter = 0
num_actions = 5 #Subject to change, depends on game
replay_size = 100 
minibatch_size = 32
memory_replay = []
init_epsilon = 0.99
end_epsilon = 0.15
epsilon_decay = 5 #this is how many games it will take for epsilon to be 0.2
gamma = 0.99
action = np.random.randint(num_actions)


# setup some basic config info
config_info = middletier.config.check() # data pulled from config.ini
# print(f"config_info {config_info['directories']['data']}")
game_path = config_info['directories']['data'] + "/NES/Excitebike.json"     # path to our game info (addreses and such)
rom_path = config_info['directories']['data'] + "/NES/roms/Excitebike (JU) [!].nes" # path to our ROM

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

#client.send_input('P1 A', state=True)
#client.send_input('P1 B', state=True)

#Actual iteration of the AI 

def next_frame(action_index, frames: int = 1): #This is for getting the next frame in the game. We may have the game do the same action for multiple frames.
    images = []
    # press (0) and then release (1) the given button
    for frame in range(frames ):
        if action == 0: 
            client.send_input('P1 Up')
        elif action == 1:
            client.send_input('P1 Down')
        elif action == 2:
            client.send_input('P1 Left')
        elif action == 3:
            client.send_input('P1 Right')
        elif action == 4:
            client.send_input('P1 A',state=True)
        #     pass

        client.conn.advance_frame()
        client.send_input('P1 A',state=False)
        #if frame % 2:
        images.append(screenshot()) # might need to turn into torch tensor

    # TODO: JUST TAKE WHAT WE NEED
    # loop through each address, read the state of the memory
    for addr in client.game.addresses:
        read_val = client.conn.read_byte(addr)
        try:
            memory_state[addr] = int(read_val.decode('ascii')[1:])
        except AttributeError:
            memory_state[addr] = 0

    speed = memory_state[memory_map['Velocity (Bike)']]

    return speed, images
    

#Frame stacking by pulling image and advancing state 4 times
reward, screenshots = next_frame(action, frames=4) #doesn't need to get reward each time, the last time is the only one that matters

prev_speed = 0
# TODO: don't do this
reward = reward - prev_speed

# [[image1],[image2],[image3],[image4]]

# state = torch.stack((screenshots[0], screenshots[1], screenshots[2], screenshots[3]),0)
state = torch.Tensor(screenshots)


# print(f"state: {state.shape}")
print(torch.cuda.is_available())

num_of_games = 50000
num_frames = 1000
prev_average = 0
#last_six_average = []
#loss_graph = []

#last_six_average.append(0.6)

# track progress over each run
episode_speed = 0
common_speed = []
runs = []
running_avg = []
# setup plotting progress
matplotlib.pyplot.ion()
fig = matplotlib.pyplot.figure()
ax = matplotlib.pyplot.subplot(1,1,1)
ax.set_xlabel('Episode')
ax.set_ylabel('Average Speed')
ax.plot(running_avg, common_speed, 'o--', markersize = 1, color='grey')
fig.show()



for episode in range(num_of_games): #The number of games we want to have it play. 
    logger.info(f"starting episode {episode}/{num_of_games}")
    episode_speed = 0
    flag = 1
    num_frames = 0
    #for i in range(num_frames):
    while flag: 
        num_frames += 1
        #print(reward)
        state = state.unsqueeze(0)
        output = myNetwork(state.to(device))
        state = state.squeeze(0)

        #Might be able to put action on the gpu as well
        if torch.cuda.is_available():
            action = torch.tensor(action).to(device)
    

        #epsilon = calc_epsilon(episode)   
        epsilon = 0.7
        
        
        #Determining whether the action is random or ideal
        if np.random.random() > epsilon:
            action = torch.argmax(output.detach().cpu()).item()
        else:
            action = np.random.randint(num_actions)
    
        
        

        #Frame stacking by pulling image and advancing state 4 times
        reward, screenshots = next_frame(action, frames=4) #doesn't need to get reward each time, the last time is the only one that matters
        
        episode_speed += reward
        
        '''
        #Image displacement reward
        reward = image_displacement(state[2],state[3])
        print(reward) '''

        # TODO: Maybe use constant to subtract speed from instead
        reward = reward - prev_speed

        # state_2 = torch.stack((screenshots[0], screenshots[1], screenshots[2], screenshots[3]),0)
        state_2 = torch.Tensor(screenshots)
        # print(f"state_2: {state_2.shape}")

        memory_replay.append((state,action,reward,state_2))

        # print("after memory_replay.append()")

        if len(memory_replay) > replay_size:
            memory_replay.pop()

        # print("after memory_replay.pop()")
    
        if len(memory_replay) >= minibatch_size:
            
            if num_frames == 32:
                Finish = state_2[3]

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
            next_q_values = TargetNetwork(state_2_batch).max(1)[0].detach()
            expected_q_values = reward_batch + gamma * next_q_values

            
            #Updates and others
            loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

            #Saving losses to show on graph
            #loss_graph.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = state_2

            #To save finish screen 
            #if i == 40:
            #    torch.save(state[3],"GameFin.pt")

            #Ends game if the last image is not similar enough to normal game image
            if torch.nn.functional.cosine_similarity(state[3].flatten(), Finish.flatten(), dim=0) < 0.7:
                flag = 0
            prev_speed = reward
            
    # print("while loop done")

    episode_average = episode_speed / num_frames
    
    print(episode_average)
    #print(max(last_six_average))
    

    #saves model
    #if episode_average >= 0.90 * max(last_six_average):
    #    print("weights updated")
    #    torch.save(myNetwork.state_dict(), 'saved_weights.pth')
    print("weights updated")
    torch.save(myNetwork.state_dict(), 'saved_weights.pth')

    #This is updating the target network every ? episodes, it is updating based off of the original network not the file of the original
    if episode % 1 == 0:
        TargetNetwork.load_state_dict(myNetwork.state_dict())
    
    #Loading the model if it has had any training done
    if os.path.isfile('saved_weights.pth'):
        print("weights loaded")
        myNetwork.load_state_dict(torch.load('saved_weights.pth'))

    #last_six_average.append(episode_average)
    #if len(last_six_average) > 1:
    #    last_six_average.pop(0)

    
    common_speed.append(episode_average)
    running_avg.append(sum(common_speed) / (episode + 1))
    runs.append(episode)
    
    
    logger.info(f"ending episode {episode}/{num_of_games}")
    
    logger.info(f"episode average: {episode_average}")
    # matplotlib.pyplot.scatter(episode, episode_average)
    # matplotlib.pyplot.pause(0.05)
    # ax.lines[0].set_data(runs, common_speed)
    ax.lines[0].set_data(runs, running_avg)
    ax.relim()  
    ax.autoscale_view() 
    
    '''
    #start loss plot 
    plt.clf()
    plt.plot(loss_graph)
    plt.xlabel('Frames')
    plt.ylabel('Loss')
    plt.title('Loss over frames')
    plt.show()
    #End loss plot
    loss_graph = []
    '''

    fig.canvas.flush_events()

    client.conn.load_state()

logger.info(f"average speed: {sum(common_speed) / len(common_speed)}")

matplotlib.pyplot.show()
while True:
    matplotlib.pyplot.pause(0.05)


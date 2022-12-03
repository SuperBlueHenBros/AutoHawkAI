import numpy as np
import middletier
import logging

# setup some basic config info
config_info = middletier.config.check() # data pulled from config.ini
game_path = config_info['directories']['data'] + "/NES/Excitebike.json"     # path to our game info (addreses and such)
rom_path = "C:/Users/Mike/Documents/Homebrew/NES Games/Excitebike (JU) [!].nes" # path to our ROM

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

#These are the actions we can take
actions = ['up','down','left','right']

#Initializing the q matrix
#Those numbers are just the range of the inputs we are using so speed could be the one that says 100. So the speed could be anywhere from 1 to 100.
#Actions in order: speed, terrain, status, lane, air, angle
q_values = np.zeros((5,6,4,9,2,11,len(actions))) #Change to new dimensions

#These are just setting up parameters we will need later
epsilon = 0.9 #percent of times we take best action
discount = 0.9 #discount factor for future rewards
learn_rate = 0.9 #rate AI learns

#function to get the next action. 90% of the time it will select the best option. The remaining 10% it will select a random action in order to explore more options.
def get_next_action(speed, terrain, status, lane, air, angle,epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[speed, terrain, status, lane, air, angle])
    else:
        return np.random.randint(len(actions))

def next_frame(action_index, frames: int = 1): #This is for getting the next frame in the game. We may have the game do the same action for multiple frames.
    # press (0) and then release (1) the given button
    for i in range(2):
        if actions[action_index] == 'up': 
            client.send_input('P1 Up')
        elif actions[action_index] == 'down':
            client.send_input('P1 Down')
        elif actions[action_index] == 'left':
            client.send_input('P1 Left')
        elif actions[action_index] == 'right':
            client.send_input('P1 Right')

        for frame in range(frames):
            client.conn.advance_frame()
        
    # loop through each address, read the state of the memory
    for addr in client.game.addresses:
        read_val = client.conn.read_byte(addr)
        try:
            memory_state[addr] = int(read_val.decode('ascii')[1:])
        except AttributeError:
            memory_state[addr] = 0

    speed = memory_state[memory_map['Velocity (Bike)']]
    terrain = memory_state[memory_map['Course Map 3']]
    status = memory_state[memory_map['Status Effect']]
    lane = memory_state[memory_map['Current Lane']]
    air = memory_state[memory_map['In Air']]
    angle = memory_state[memory_map['Bike Angle']]

    speed, terrain, status, lane, air, angle = convert_values(speed, terrain, status, lane, air, angle)

    return speed, terrain, status, lane, air, angle

def convert_values(speed, terrain, status, lane, air, angle):
    
    #This is handling the conversion of terrain               
    if terrain == 42:
        terrain = 0
    elif terrain == 59 or terrain == 61: #Nothing
        terrain = 1
    elif terrain == 112 or terrain == 114 or terrain == 185 or terrain == 187 or terrain == 188 or terrain == 190: #Mud
        terrain = 2
    elif terrain > 226 and terrain < 232:
        terrain = 3
    elif terrain == 136 or terrain == 137: #Boost
        terrain = 4
    else:
        terrain = 5



    #Conversion of status
    if status == 4:
        status = 3

    #Lanes
    if lane < 14:
        lane = 0
    if lane == 14:
        lane = 1
    elif lane > 14 and lane < 26:
        lane = 2
    elif lane == 26:
        lane = 3
    elif lane > 26 and lane < 38:
        lane = 4
    elif lane == 38:
        lane = 5
    elif lane > 38 and lane < 50:
        lane = 6
    elif lane == 50:
        lane = 7
    elif lane > 50:
        lane = 8
    
    #Air conversion
    if air == 0:
        air = 0
    else:
        air = 1

    #Angle conversion
    if angle == 2:
        angle = 0
    elif angle == 3:
        angle = 1
    elif angle == 4:
        angle = 2
    elif angle == 5:
        angle = 3
    elif angle == 6:
        angle = 4
    elif angle == 7:
        angle = 5
    elif angle == 8:
        angle = 6
    elif angle == 9:
        angle = 7
    elif angle == 10:
        angle = 8
    elif angle == 11:
        angle = 9
    else:
        angle = 10
    
    return speed, terrain, status, lane, air, angle

num_of_games = 100

client.send_input('P1 A', state=True)
client.send_input('P1 B', state=True)

client.conn.save_state()

for episode in range(num_of_games): #The number of games we want to have it play. 
    print("starting episode")

    action_index = 0
    speed, terrain, status, lane, air, angle  = next_frame(action_index) #This is setting those variables equal to the values from the game

    # speed, terrain, status, lane, air, angle  = np.random.randint(4), np.random.randint(6),np.random.randint(4),np.random.randint(6),np.random.randint(2),np.random.randint(11)


    # while not game_over: #While the current race is still going
    for i in range(550):  
        action_index = get_next_action(speed, terrain, status, lane, air, angle, epsilon) 

        old_speed, old_terrain, old_status, old_lane, old_air, old_angle = speed, terrain, status, lane, air, angle

        speed, terrain, status, lane, air, angle = next_frame(action_index, 5) #This line updates the parameters after passing the desired action and getting the next frame

        reward = speed - 3 # changed from 3

        old_q_value = q_values[old_speed, old_terrain, old_status, old_lane, old_air, old_angle, action_index]
        
        temporal_difference = reward + (discount * np.max(q_values[speed, terrain, status, lane, air, angle])) - old_q_value

        new_q_value = old_q_value + (learn_rate * temporal_difference)
        q_values[old_speed, old_terrain, old_status, old_lane, old_air, old_angle, action_index] = new_q_value

    client.conn.load_state()
    print("ending episode")
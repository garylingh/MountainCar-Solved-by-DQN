#!/usr/bin/python3
# Mountain Car Hand input with Reward

import gym
import numpy as np
from numpy import savetxt
from datetime import datetime
import sys,tty,termios

def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def getkey():
        #inkey = _Getch()
        while(1):
            k = getch()
            if k != '':
                break
            
        if k=='\x1b[A':   # UP
            return 3
        elif k=='\x1b[B': # Down
            return 1
        elif k=='\x1b[C': # Right
            return 2
        elif k=='\x1b[D': # Left
            return 0
        else:
            return 9
        
def showKey(key):
    if key == 0:
        return "Left"
    elif key == 1:
        return "Down"
    elif key == 2:
        return "Right"
    elif key == 3:
        return "Up"

def saveReward(Q):

    f = open(filename + ".reward", "w")
    for i, x in enumerate(Q, start=1):
        new_state, action, reward, rewards, state = x
        line = str(i) + "," + showKey(action)+ "," + str(new_state) + "," + str(reward) + "," + str(rewards)
        f.write(line+"\n")
    f.close()


def main():
    env   = gym.make("MountainCar-v0")
    state = env.reset()

    done, step, rewards, Q = False, 0, 0, []
    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    while not done:
        step += 1
        env.render()
        
        action = getkey()
        if action not in [0,1,2]:
            break
            
        new_state, _, done, _ = env.step(action)
        
        position = round(state[0],2)
        speed = round(state[1], 3)
        new_position = round(new_state[0],2)
        new_speed = round(new_state[1], 3)
        
        reward = (new_position-position) * new_speed
        rewards += reward
        Q.append([np.array([new_position, new_speed]), action, reward, rewards, np.array([position, speed])])
        state = new_state

        #print("Step:", step, ", Act:", showKey(action), ", State:", new_state, "R:", reward, ", R Sum:", rewards)
        print(step, ",", showKey(action), ",", new_state, ",", reward, ",", rewards)
    
    env.close()
    
    savetxt(filename + '.Q',Q)
    saveReward(Q)

if __name__ == "__main__":
    main()
#!/usr/bin/python3

# Mountain Car Hand input with Reward

import gym
import numpy as np
from numpy import save, load
from datetime import datetime
import sys,tty,termios

def showKey(key):
    if key == 0:
        return "Left"
    elif key == 1:
        return "Down"
    elif key == 2:
        return "Right"
    elif key == 3:
        return "Up"

def intState(state):
    position = int(round(state[0],2) * 100) + 120
    speed = int(round(state[1],3) * 1000) + 70
    return position, speed
    
def main():
    env   = gym.make("MountainCar-v0")
    state = env.reset()
    step = 0

    filename = input("Please enter trained Q-table:")
    Q = load(filename,allow_pickle=True)

    loops_enter = input("Total loops to run: (Defalut " + str(1000) +  ") Total: ")
    loops = 1000 if loops_enter == "" else int(loops_enter)
    showRender = input("Show running animation? (y/n)")

    while step < loops:
        step += 1
        if showRender == 'y':
            env.render()

        # Choose action from Q table
        Q_pos, Q_spd = intState(state)
        action = np.argmax(Q[Q_pos, Q_spd,:] + np.random.randn(1,3)*(1./(step+1)))
        state, _, _, _ = env.step(action)

        print(step, ",", showKey(action), ",", Q_pos, Q_spd, ",", Q[Q_pos, Q_spd,action])

        if state[0] > 0.5:
            if input("Run again?(y/n)") == 'y':
                step = 0
                env.reset()
            else:
                break
    env.close()

if __name__ == "__main__":
    main()

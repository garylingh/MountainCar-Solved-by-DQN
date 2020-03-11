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

def saveReward(Q, step):
    filename = "Q_" + str(step) + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
    save(filename, Q, allow_pickle=True)

def intState(state):
    position = int(round(state[0],2) * 100) + 120
    speed = int(round(state[1],3) * 1000) + 70
    return position, speed

def printResult(step, action, Qnew_p, Qnew_s, reward):
    print(step, ",", showKey(action), ",", Qnew_p, Qnew_s, ",", reward)

def main():
    env   = gym.make("MountainCar-v0")
    state = env.reset()
    
    eta = .628
    gma = .9

    positions  = 181 # -1.2 to 0.6 with 0.01 per step
    speeds     = 141 # -0.7 to 0.7 with 0.01 per step
    
    action_spaces = env.action_space.n
    Q = np.zeros([positions, speeds, action_spaces])
    loops=1000
    done, step = False, 0
    loops_enter = input("Total loops to run: (Defalut " + str(loops) +  ") Total: ")
    loops = 1000 if loops_enter == "" else int(loops_enter)

    showRender = input("Show running animation? (y/n)") == 'y'
    showText = input("Show running Text result? (y/n)") == 'y'
    load_Q = input("Load Pre-trained Q-table? (y/n)")=='y'

    if load_Q:
        Q_file = input("Please enter Q-table name: ")
        Q = load(Q_file, allow_pickle=True)

    while step < loops:
        step += 1
        if showRender:
            env.render()
        
        Q_pos, Q_spd = intState(state)

        # Choose action from Q table
        action = np.argmax(Q[Q_pos, Q_spd,:] + np.random.randn(1, 3)*(1./(step+1)))
        new_state, _, done, _ = env.step(action)

        Qnew_p, Qnew_s = intState(new_state)
        reward = (Qnew_p - Q_pos) * (Qnew_s-Q_spd)

        #Update Q-Table with new knowledge
        Q[Q_pos,Q_spd,action] = Q[Q_pos,Q_spd,action] + eta * (reward + gma * np.max(Q[Qnew_p,Qnew_s,:]) - Q[Q_pos,Q_spd,action])
        
        state = new_state

        if showText:
            #print(step, ",", showKey(action), ",", Qnew_p, Qnew_s, ",", reward)
            printResult(step, action, Qnew_p, Qnew_s, reward)

        if new_state[0] > 0.5:
            env.reset()
        
    env.close()
    saveReward(Q,step)
    printResult(step, action, Qnew_p, Qnew_s, reward)

if __name__ == "__main__":
    main()

#!/usr/bin/python3

import gym
import numpy as np
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

def main():
    env   = gym.make("MountainCar-v0")
    state = env.reset()

    done, step = False, 0
    
    while not done:
        step += 1
        env.render()
        
        action = getkey()
        if action not in [0,1,2]:
            break
            
        new_state, reward, done, _ = env.step(action)
        print("Step:", step, ", Action:", action, ", State:", new_state)

if __name__ == "__main__":
    main()
# MountainCar-Solved-by-DQN

The "Reward" from Original OpenAI GYM MountCar-v0 "Environment" is useless.  You can Not learn from it.  To let then the Agent learns and builds the Q-table, we must re-define and create a new reward system.

By MountainCar-v0 definitions:
## Obervation
Num | Observation | Min | Max
--:|:--|--:|--:
0| Car Position | -1.2 | 0.6
1| Car Velocity | -007 | 0.07

## Actions
 Value | Actions
 --:|:--
 0 | Push Left
 1 | No Push
 2 | Push Right
 
## Q-table definition
* Car Position: 
  * From: -1.2 / 0.01 = -120
  * Plus: 0
  * To: 0.6 / 0.01 = 60
  * Total: 120 + 1 + 60 = 181
* Car Velocity:
  * From -0.07 / 0.0001 = -70
  * Plus: 0
  * To: 0.07 / 0.0001 = 70
  * Total: 70 + 1 + 70 = 141
* Q table
```python
import numpy as np

Q = np.zeros(181,141,3) # (position, velocity, actions)
```
## Initial State
* Starting position: -0.6 to -0.4

## Goal
* Reach position: 0.5

## Reward definition
Here is the key point.  How do we define the reward for Q table to learn over training.

```python
reward = (Qnew_p - Q_pos) * (Qnew_s - Q_spd)
```
* movement = new_Position - old_position
* change of speed = new_Velocity - old_velocity

|||Position|||Velocity||Reward|
|--:|--:|--:|--:|--:|--:|--:|--:|
||Current | New | Movement | Current | New | Change | Reward|
|Move Left|-0.5|-0.6|-0.1|-0.02|-0.01|0.01|-0.001000|
|Move Left|-0.7|-0.8| 0.1|-0.04|-0.03|0.01|-0.001000|
|Move Left| 0.3 |0.2 |-0.1|0.02| 0.03|0.01|-0.001000|
|Move Right|-0.8|-0.7| 0.1|0.05| 0.06|0.01| 0.001000|
|Move Right|-1.2|-1.1| 0.1|0.01| 0.02|0.01| 0.001000|
|Move Right| 0.3| 0.4| 0.1|0.02| 0.01|-0.01| -0.001000|

* Move toward Left gets Negative Reward 
* Move toward Right gets Positive Reward

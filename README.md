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
  

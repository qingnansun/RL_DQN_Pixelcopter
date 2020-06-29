# RL_DQN_Pixelcopter
## Project for Baidu RL course

This project employed the DQN algorithm from [PARL](https://github.com/PaddlePaddle/PARL) of Baidu.

All the needed funcions were written in the single python file, in which:
- The most functions, e.g. Model(), Agent(), ReplayMemory(), as well as part of main(), etc. are indentical as or were slightly modified based on the materials provided by the [Baidu RL course](https://aistudio.baidu.com/aistudio/education/group/info/1335).

- The preprocessing (scaling) of the state was inspired by [nbuliyang's project](https://github.com/nbuliyang/RL)

- The needed libraries and corresponding versions are documented in requirements.txt

## Results
The figures below shows the test_reward (mean value of 5 test episodes) and the max_reward (the maximum value among the 5 test episodes).
![The first episodes](./assets/1_first_episodes.png)
![The first episodes](./assets/2_3000plus_episodes.png)

Q-learning in Maze World<a name="TOP"></a>
===================

This project uses Q-learning Implementation to guide robot through a 10x10 Maze. 

I have used Epsilon Greedy Exploration policy, where a robot explores it's environment during the starting phase of training 
and focuses more onto exploiting the reward towards the end of training

---

The code for this game runs in python 

1. Pygame needs to be installed to run this program with environment. command for installing pygame is 
   - pip install pygame


2. Run Train.py from code. It generates random maze sequences. If a maze sequence with goal is blocked by obstacles, It reruns the code to generate new sequence


3. You can run the code test.py multiple times to check the policy working for robot from different locations. Each time the program is run robot starts from a random position

---

Demos <a name="TOP"></a>
===================

![1](https://user-images.githubusercontent.com/58717184/108563059-d8240200-72ce-11eb-8144-2e2cc67529bb.gif)


![2](https://user-images.githubusercontent.com/58717184/108563110-eeca5900-72ce-11eb-9869-07f0818d1d30.gif)



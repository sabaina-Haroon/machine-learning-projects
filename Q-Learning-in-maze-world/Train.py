import numpy as np
import pygame
from time import time, sleep
from random import randint as r
import random
from matplotlib import pyplot as plt

n = 10  # represents no. of side squares(n*n total squares)
scrx = n * 100
scry = n * 100
background = (51, 51, 51)  # used to clear screen while rendering
screen = pygame.display.set_mode((scrx, scry))  # creating a screen using Pygame
colors = [(51, 51, 51) for i in range(n ** 2)]
reward = np.zeros((n, n))
terminals = []
penalities = 25
obstacles = []

# generating 25 obstacle blocks/walls at random locations in maze
# defining reward against such obstacles
while penalities != 0:
    i = r(0, n - 1)
    j = r(0, n - 1)
    obstacles.append([i, j])
    if reward[i, j] == 0 and [i, j] != [0, 0] and [i, j] != [n - 1, n - 1]:
        reward[i, j] = -100
        penalities -= 1
        colors[n * i + j] = (255, 0, 0)
        terminals.append(n * i + j)
reward[n - 1, n - 1] = 100          #reward for goal state
colors[n ** 2 - 1] = (0, 255, 0)
terminals.append(n ** 2 - 1)
obstacles.append([n-1, n-1])

Q = np.zeros((n ** 2, 4))  # Initializing Q-Table
actions = {"up": 0, "down": 1, "left": 2, "right": 3}  # possible actions
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i, j)] = k
        k += 1
alpha = 0.01
gamma = 0.9
current_pos = [0, 0]
epsilon = 1


# when an episode finishes, start robot from a random location
def random_starting_pt():
    i = r(0, n - 1)
    j = r(0, n - 1)
    new_pt = [i, j]
    if new_pt not in obstacles:
        return new_pt
    else:
        while new_pt in obstacles:
            i = r(0, n - 1)
            j = r(0, n - 1)
            new_pt = [i, j]
        return new_pt


# select action based on epsilon greedy strategy
def select_action(current_state):
    global current_pos, epsilon
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n - 1:
            possible_actions.append("down")
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n - 1:
            possible_actions.append("right")
        action = actions[possible_actions[r(0, len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0:  # up
            possible_actions.append(Q[current_state, 0])
        else:
            possible_actions.append(m - 100)                    # this stops robot from going out of maze
        if current_pos[0] != n - 1:  # down
            possible_actions.append(Q[current_state, 1])
        else:
            possible_actions.append(m - 100)                     # this stops robot from going out of maze
        if current_pos[1] != 0:  # left
            possible_actions.append(Q[current_state, 2])
        else:
            possible_actions.append(m - 100)                     # this stops robot from going out of maze
        if current_pos[1] != n - 1:  # right
            possible_actions.append(Q[current_state, 3])
        else:
            possible_actions.append(m - 100)                     # this stops robot from going out of maze

        # randomly selecting one of all possible actions with maximin value
        action = random.choice([i for i, a in enumerate(possible_actions) if a == max(
            possible_actions)])
    return action


def transition(action):
    chance = random.uniform(0.0, 1.0)

    # let robot where it wants to go with 60 percent probability and randomly explore other 40 percent times.
    if chance < 0.6:
        action = action
    elif chance < 0.7:
        action = (action + 1) % 4
    elif chance < 0.8:
        action = (action + 2) % 4
    elif chance < 0.9:
        action = (action + 3) % 4
    else:
        return current_pos, action  # not moving/ not taking any action

    if (action == 0) and (current_pos[0] != 0):  # move up
        current_pos[0] -= 1
    elif (action == 1) and (current_pos[0] != (n - 1)):  # move down
        current_pos[0] += 1
    elif (action == 2) and (current_pos[1] != 0):  # move left
        current_pos[1] -= 1
    elif (action == 3) and (current_pos[1] != (n - 1)):  # move right
        current_pos[1] += 1

    return current_pos, action


def episode():
    step_size = 0
    hit_wall = False
    reached_goal = False
    global current_pos, epsilon

    while (step_size < 150) and (not reached_goal):
        # sleep(0.5)
        screen.fill(background)
        layout()
        pygame.display.flip()

        current_state = states[(current_pos[0], current_pos[1])]

        # select an action based on exploration policy
        action = select_action(current_state)

        # find new state from transition function
        current_pos, action = transition(action)
        new_state = states[(current_pos[0], current_pos[1])]

        # use full form of bellman equation if next step doesn't take robot to obstacle state else use simplified
        # version of the equation
        if new_state not in terminals:
            Q[current_state, action] += alpha * (
                    reward[current_pos[0], current_pos[1]] + gamma * (np.max(Q[new_state])) - Q[current_state, action])
        else:
            Q[current_state, action] += alpha * (reward[current_pos[0], current_pos[1]] - Q[current_state, action])

            # reducing as time increases to satisfy Exploration & Exploitation Trade-off
            if epsilon > 0.000001:
                epsilon -= 3e-4
            if reward[current_pos[0], current_pos[1]] == 100:
                reached_goal = True
            current_pos = random_starting_pt()

        step_size += 1

    print(step_size)


# generates graphics layout for maze board
def layout():
    c = 0
    for i in range(0, scrx, 100):
        for j in range(0, scry, 100):
            pygame.draw.rect(screen, (255, 255, 255), (j, i, j + 100, i + 100), 0)
            pygame.draw.rect(screen, colors[c], (j + 3, i + 3, j + 95, i + 95), 0)
            c += 1
            pygame.draw.circle(screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)



# main loop for training
run = True
no_of_episodes = 0
while no_of_episodes != 600:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    episode()
    no_of_episodes += 1
    print('epsilon', epsilon)
    if epsilon < 0.000001:
        np.save('Q_table', Q)
        np.savetxt('Q_excel.csv', Q, delimiter=',', fmt='%d')
        np.save('colors', colors)
        np.save('rewards', reward)
        np.save('terminals', terminals)
        print('q_table', Q)
        print(no_of_episodes, 'episode no')
        run = False
pygame.quit()

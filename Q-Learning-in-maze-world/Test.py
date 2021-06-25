import numpy as np
import pygame
from time import time, sleep
from random import randint as r
import random

n = 10  # represents no. of side squares(n*n total squares)
scrx = n * 100
scry = n * 100
background = (51, 51, 51)  # used to clear screen while rendering
screen = pygame.display.set_mode((scrx, scry))  # creating a screen using Pygame
colors = np.load('colors.npy')
reward = np.load('rewards.npy')
terminals = np.load('terminals.npy')

Q = np.load('Q_table.npy')  # LOADING Q-Table
actions = {0: "up", 1: "down", 2: "left", 3: "right"}  # possible actions
actions_taken = []
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i, j)] = k
        k += 1
alpha = 0.01
gamma = 0.9

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


obst = np.transpose(np.nonzero(reward))
obstacles = []
for item in obst:
    obstacles.append([item[0], item[1]])

current_pos = random_starting_pt()


def select_action(current_state):
    global current_pos, epsilon
    possible_actions = []
    m = np.min(Q[current_state])
    if current_pos[0] != 0:  # up
        possible_actions.append(Q[current_state, 0])
    else:
        possible_actions.append(m - 100)
    if current_pos[0] != n - 1:  # down
        possible_actions.append(Q[current_state, 1])
    else:
        possible_actions.append(m - 100)
    if current_pos[1] != 0:  # left
        possible_actions.append(Q[current_state, 2])
    else:
        possible_actions.append(m - 100)
    if current_pos[1] != n - 1:  # right
        possible_actions.append(Q[current_state, 3])
    else:
        possible_actions.append(m - 100)
    action = random.choice([i for i, a in enumerate(possible_actions) if a == max(
        possible_actions)])  # randomly selecting one of all possible actions with maximin value

    return action


def episode():
    step_size = 0
    hit_wall = False
    reached_goal = False
    global current_pos, epsilon

    while (step_size < 150) and (not reached_goal):
        sleep(0.5)
        screen.fill(background)
        layout()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        pygame.display.flip()
        current_state = states[(current_pos[0], current_pos[1])]
        action = select_action(current_state)
        actions_taken.append(actions[action])
        if action == 0:  # move up
            current_pos[0] -= 1
        elif action == 1:  # move down
            current_pos[0] += 1
        elif action == 2:  # move left
            current_pos[1] -= 1
        elif action == 3:  # move right
            current_pos[1] += 1
        new_state = states[(current_pos[0], current_pos[1])]
        if new_state in terminals:
            if reward[current_pos[0], current_pos[1]] == 100:
                sleep(0.5)
                reached_goal = True
                layout()
                pygame.display.flip()
                sleep(1.0)
                return reached_goal
        step_size += 1


def layout():
    c = 0
    for i in range(0, scrx, 100):
        for j in range(0, scry, 100):
            pygame.draw.rect(screen, (255, 255, 255), (j, i, j + 100, i + 100), 0)
            pygame.draw.rect(screen, colors[c], (j + 3, i + 3, j + 95, i + 95), 0)
            c += 1
            pygame.draw.circle(screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)


run = True
reached_goal = False
while not reached_goal:
    reached_goal = episode()
    if reached_goal:
        print(actions_taken)

pygame.quit()

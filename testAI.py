'''
Refrences -
    1. https://www.gamasutra.com/view/feature/132330/the_pacman_dossier.php?page=1
    2. https://gameinternals.com/understanding-pac-man-ghost-behavior

This is an attempt to recreate pacman in pygame.

AUTHORS -
    1. Rohan Singh
    2. Maruf Hussain
    3. Tarun Singh Tomar

This version of pacman will be used to implement
a genetic algorithm which will be able to play
pacman.

A project by Robotics Club IIT Jodhpur.

'''

__version__ = '0.23'


import pygame
import time
from utils import *

# from entities import *

# ---------------------------------------------------------- Entities --------------------------------------------------------

'''
The classes module.

Contains Wall and Path tile blocks.

Contains the pacman class.

A ghost class.

Classes for each of the ghosts in pacman,
Blinky, Inky, Pinky, Clyde.
Each ghost class inherits the generic Ghost class.
'''


# The wall class


class Wall():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinate = (x, y)


class Path():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinate = (x, y)
# ------------------------------------------------------- Pacman Class --------------------------------------------------


class Pacman():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinate = (x, y)
        self.direction = (1, 0)
        self.next = get_block(self.coordinate, self.direction)
        self.sprite = pacman_l
        self.mouth_open = False
        self.tmpdirection = (1, 0)

    # Make a function to update pacman

    def update(self):

        # Pacman Sprite Update
        if self.mouth_open:
            self.sprite = pacman_c
        else:
            if self.direction == (1, 0):
                self.sprite = pacman_l
            if self.direction == (-1, 0):
                self.sprite = pacman_r
            if self.direction == (0, 1):
                self.sprite = pacman_d
            if self.direction == (0, -1):
                self.sprite = pacman_u
        self.mouth_open = ~(self.mouth_open)

        i, j = self.next
        if maze[j][i] == 0:
            self.coordinate = get_block(self.coordinate, self.direction)
        else:
            i, j = get_block(self.coordinate, self.tmpdirection)
            if maze[j][i] == 0:
                self.direction = self.tmpdirection
                self.coordinate = get_block(self.coordinate, self.direction)

        self.next = get_block(self.coordinate, self.direction)
        # screen.blit(self.sprite, coor_to_px(self.coordinate))

    def draw(self):
        screen.blit(self.sprite, coor_to_px(self.coordinate))

    def type_node(self):
        poss = [False, False, False, False]

        i, j = self.coordinate

        if maze[j - 1][i] == 0:
            poss[0] = True
        if maze[j][i + 1] == 0:
            poss[1] = True
        if maze[j + 1][i] == 0:
            poss[2] = True
        if maze[j][i - 1] == 0:
            poss[3] = True
        return poss


# ------------------------------------------------------- Ghost Classes --------------------------------------------------
# Ghost Class
class Ghost():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinate = (x, y)  # (29,27) by default
        self.left = None
        self.right = None
        self.direction = (0, 1)
        self.target = pacman.coordinate
        self.sprite = blinky_1_l
        self.phase_1 = False
        self.mode = 'chase'
        self.home = (2, 3)
        self.counter = 1
        self.threshold = get_threshold(self.counter)

    def draw(self):
        screen.blit(self.sprite, coor_to_px(self.coordinate))

    def getpos(self):
        x, y = self.direction
        self.left = (y, x)
        self.right = (-1*y, -1*x)

    def type_node(self):
        poss = []
        i, j = get_block(self.coordinate, self.direction)
        if maze[j][i] == 0:
            poss.append(self.direction)
        i, j = get_block(self.coordinate, self.left)
        if maze[j][i] == 0:
            poss.append(self.left)
        i, j = get_block(self.coordinate, self.right)
        if maze[j][i] == 0:
            poss.append(self.right)
        # print(poss)
        return poss

    def update(self):
        # Sprite Update
        if self.mode == 'chase':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = blinky_1_r
                if self.direction == (-1, 0):
                    self.sprite = blinky_1_l
                if self.direction == (0, 1):
                    self.sprite = blinky_1_d
                if self.direction == (0, -1):
                    self.sprite = blinky_1_u
            else:
                if self.direction == (1, 0):
                    self.sprite = blinky_2_r
                if self.direction == (-1, 0):
                    self.sprite = blinky_2_l
                if self.direction == (0, 1):
                    self.sprite = blinky_2_d
                if self.direction == (0, -1):
                    self.sprite = blinky_2_u
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        if self.mode == 'scatter':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            else:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        self.counter += 1

        if self.counter == self.threshold:
            if get_threshold(self.counter) == 0:
                self.mode = 'chase'
                self.choose_target_tile()
                self.threshold += get_threshold(self.counter)
            else:
                #print("This code was accessed")
                self.threshold += get_threshold(self.counter)
                if self.mode == 'chase':
                    self.mode = 'scatter'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif self.mode == 'scatter':
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif get_threshold(self.counter) == 0:
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)

    def choose_target_tile(self):
        if self.mode == 'chase':
            self.find_target()
        if self.mode == 'scatter':
            self.target = self.home

    def set_home(self, coor_tuple):
        self.home = coor_tuple

    def find_target(self):
        self.target = pacman.coordinate


# ------------------------------------------------------- Blinky Class --------------------------------------------------
class Blinky(Ghost):
    pass


# ------------------------------------------------------- Inky Class --------------------------------------------------
class Inky(Ghost):
    def update(self):
        self.find_target()
        # Sprite Update
        if self.mode == 'chase':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = inky_1_r
                if self.direction == (-1, 0):
                    self.sprite = inky_1_l
                if self.direction == (0, 1):
                    self.sprite = inky_1_d
                if self.direction == (0, -1):
                    self.sprite = inky_1_u
            else:
                if self.direction == (1, 0):
                    self.sprite = inky_2_r
                if self.direction == (-1, 0):
                    self.sprite = inky_2_l
                if self.direction == (0, 1):
                    self.sprite = inky_2_d
                if self.direction == (0, -1):
                    self.sprite = inky_2_u
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        if self.mode == 'scatter':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            else:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        self.counter += 1

        if self.counter == self.threshold:
            if get_threshold(self.counter) == 0:
                self.mode = 'chase'
                self.choose_target_tile()
                self.threshold += get_threshold(self.counter)
            else:
                self.threshold += get_threshold(self.counter)
                if self.mode == 'chase':
                    self.mode = 'scatter'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif self.mode == 'scatter':
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif get_threshold(self.counter) == 0:
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)

    def find_target(self):
        vector = get_vector(pacman.coordinate, blinky.coordinate)
        pac_ahead = get_block(pacman.coordinate, pacman.direction)
        pac_ahead = get_block(pac_ahead, pacman.direction)
        self.target = get_block(pac_ahead, vector)


# ------------------------------------------------------- Pinky Class --------------------------------------------------
class Pinky(Ghost):
    def update(self):
        # Sprite Update
        if self.mode == 'chase':
            self.find_target()
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = pinky_1_r
                if self.direction == (-1, 0):
                    self.sprite = pinky_1_l
                if self.direction == (0, 1):
                    self.sprite = pinky_1_d
                if self.direction == (0, -1):
                    self.sprite = pinky_1_u
            else:
                if self.direction == (1, 0):
                    self.sprite = pinky_2_r
                if self.direction == (-1, 0):
                    self.sprite = pinky_2_l
                if self.direction == (0, 1):
                    self.sprite = pinky_2_d
                if self.direction == (0, -1):
                    self.sprite = pinky_2_u
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        if self.mode == 'scatter':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            else:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        self.counter += 1

        if self.counter == self.threshold:
            if get_threshold(self.counter) == 0:
                self.mode = 'chase'
                self.choose_target_tile()
                self.threshold += get_threshold(self.counter)
            else:
                self.threshold += get_threshold(self.counter)
                if self.mode == 'chase':
                    self.mode = 'scatter'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif self.mode == 'scatter':
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif get_threshold(self.counter) == 0:
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)

    def find_target(self):
        # i, j = get_block(pacman.coordinate, pacman.direction)
        # self.target = pacman.coordinate
        # a = 0
        # while maze[j][i] == 0 and a < 4:
        #     self.target = get_block(self.target, pacman.direction)
        #     i, j = get_block(self.target, self.direction)
        #     a += 1
        self.target = get_block(pacman.coordinate, pacman.direction)
        self.target = get_block(self.target, pacman.direction)
        self.target = get_block(self.target, pacman.direction)
        self.target = get_block(self.target, pacman.direction)


# ------------------------------------------------------- Clyde Class --------------------------------------------------
class Clyde(Ghost):
    def update(self):
        # Sprite Update
        if self.mode == 'chase':
            self.find_target()
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = clyde_1_r
                if self.direction == (-1, 0):
                    self.sprite = clyde_1_l
                if self.direction == (0, 1):
                    self.sprite = clyde_1_d
                if self.direction == (0, -1):
                    self.sprite = clyde_1_u
            else:
                if self.direction == (1, 0):
                    self.sprite = clyde_2_r
                if self.direction == (-1, 0):
                    self.sprite = clyde_2_l
                if self.direction == (0, 1):
                    self.sprite = clyde_2_d
                if self.direction == (0, -1):
                    self.sprite = clyde_2_u
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        if self.mode == 'scatter':
            if self.phase_1:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            else:
                if self.direction == (1, 0):
                    self.sprite = scared_1_b
                if self.direction == (-1, 0):
                    self.sprite = scared_1_b
                if self.direction == (0, 1):
                    self.sprite = scared_1_b
                if self.direction == (0, -1):
                    self.sprite = scared_1_b
            self.phase_1 = ~(self.phase_1)

            self.getpos()
            poss = self.type_node()
            if(len(poss) == 1):
                self.coordinate = get_block(self.coordinate, poss[0])
                self.direction = poss[0]
            elif (len(poss) >= 2):
                dist = 100000000
                for pos in poss:
                    if dist > distance(get_block(self.coordinate, pos), self.target):
                        dist = distance(
                            get_block(self.coordinate, pos), self.target)
                        self.direction = pos
                self.coordinate = get_block(self.coordinate, self.direction)

        self.counter += 1

        if self.counter == self.threshold:
            if get_threshold(self.counter) == 0:
                self.mode = 'chase'
                self.choose_target_tile()
                self.threshold += get_threshold(self.counter)
            else:
                self.threshold += get_threshold(self.counter)
                if self.mode == 'chase':
                    self.mode = 'scatter'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif self.mode == 'scatter':
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)
                elif get_threshold(self.counter) == 0:
                    self.mode = 'chase'
                    self.choose_target_tile()
                    self.direction = change_direction(self.direction)

    def find_target(self):
        if distance(self.coordinate, pacman.coordinate) >= 8:
            self.target = pacman.coordinate
        else:
            self.target = self.home


# # Initialise characters
# pacman = Pacman(13, 19)
# pac_upd = 0
# blinky = Blinky(13, 11)
# inky = Inky(13, 7)
# inky.set_home((24, 3))
# pinky = Pinky(3, 22)
# pinky.set_home((2, 26))
# clyde = Clyde(23, 22)
# clyde.set_home((24, 26))

# entities = [pacman, blinky, inky, pinky, clyde]
# # entities = [pacman, inky]

# --------------------------------------------------------- /Entities ----------------------------------------------------------


# --------------------------------------------------- Main while loop ---------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


# Calculate the inputs at each  game loop
def generate_input():
    # print("Inputs were generated")
    inp = np.array([
        pacman.x / maze_x,
        pacman.y / maze_y,
        blinky.x / maze_x,
        blinky.y / maze_y,
        inky.x / maze_x,
        inky.y / maze_y,
        pinky.x / maze_x,
        pinky.y / maze_y,
        clyde.x / maze_x,
        clyde.y / maze_y,
        1])
    # print('input array shape : ', inp.shape)
    # print('input array : ', '\n', inp)
    return inp


# Generate the output from the inputs and the wights and biases
def propagate(ind_weights):

    # print("Propagation Started")
    z = np.matmul(generate_input(), ind_weights)
    a = sigmoid(z)

    # print('output shape : ', a.shape)
    return a


# A function to interpret the results of the NN.
def interpret(output, possiblities):

    global pacman

    dir_not_chng = True

    # print("Starting interpretation")
    while dir_not_chng:
        # print("Checking Start", '\n')
        # print(possiblities)
        # print(output)
        index = np.where(output == np.max(output))[0][0]
        value = output[index]
        # print('value and index : ', value, index)
        # Turn up
        if index == 0 and possiblities[index]:
            dir_not_chng = False
            i, j = pacman.coordinate
            j -= 1
            if maze[j][i] == 0:
                pacman.direction = (0, -1)

        # Turn right
        elif index == 1 and possiblities[index]:
            dir_not_chng = False
            i, j = pacman.coordinate
            i += 1
            if maze[j][i] == 0:
                pacman.direction = (1, 0)

        # Turn down
        elif index == 2 and possiblities[index]:
            dir_not_chng = False
            i, j = pacman.coordinate
            j += 1
            if maze[j][i] == 0:
                pacman.direction = (0, 1)

        # Turn left
        elif index == 3 and possiblities[index]:
            dir_not_chng = False
            i, j = pacman.coordinate
            i -= 1
            if maze[j][i] == 0:
                pacman.direction = (-1, 0)

        else:
            # Take the smallest possible value
            output[index] = -1


# Main while event loop
def cal_ind_fitness(ind_weights):
    global pac_upd, event, entity, pacman, maze
    running = True
    fitness = 0
    while running:
        for entity in entities[1:]:
            if pacman.coordinate == entity.coordinate:
                running = False

        for ghost in entities[1:]:
            if ghost.target == ghost.coordinate:
                ghost.choose_target_tile()

        for entity in entities:
            if entity.coordinate == (23, 13):
                entity.coordinate = (4, 13)
            elif entity.coordinate == (3, 13):
                entity.coordinate = (22, 13)

        # RGB = Red, Green, Blue
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Propagate through weights and take decision based on output.

        interpret(propagate(ind_weights), pacman.type_node())

        create_maze()

        if pac_upd == 30:
            for entity in entities:
                entity.update()
                for entity in entities[1:]:
                    if pacman.coordinate == entity.coordinate:
                        running = False
            pac_upd = 0

        pac_upd += 1

        for entity in entities:
            entity.draw()
        pygame.display.update()

        fitness += 1

    return fitness

def cal_ind_fitness_final(ind_weights):
    global pac_upd, event, entity, pacman, maze
    running = True
    fitness = 0
    while running:
        for entity in entities[1:]:
            if pacman.coordinate == entity.coordinate:
                running = False

        for ghost in entities[1:]:
            if ghost.target == ghost.coordinate:
                ghost.choose_target_tile()

        for entity in entities:
            if entity.coordinate == (23, 13):
                entity.coordinate = (4, 13)
            elif entity.coordinate == (3, 13):
                entity.coordinate = (22, 13)

        # RGB = Red, Green, Blue
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Propagate through weights and take decision based on output.

        interpret(propagate(ind_weights), pacman.type_node())

        create_maze()

        if pac_upd == 30:
            for entity in entities:
                entity.update()
                for entity in entities[1:]:
                    if pacman.coordinate == entity.coordinate:
                        running = False
            pac_upd = 0

        pac_upd += 1

        for entity in entities:
            entity.draw()
        pygame.display.update()

        fitness += 1

    return fitness


# Function to select the best parents


def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1], pop.shape[2]))
    # print("Parent Shape",parents.shape)
    for num in range(num_parents):
        max_fitness = np.where(fitness == np.max(fitness))
        max_fitness = max_fitness[0][0]
        # print("Parents",parents)
        # print("Population",pop)
        parents[num, :] = pop[max_fitness, :]
        fitness[max_fitness] = -9999999999
    return parents

#CrossOver Function
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

#Mutation
def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

# ------------------------------------------------------ /Main while loop ------------------------------------------------------


no_generations = 15
no_inputs = 10
no_outputs = 4
ind_per_pop = 24
num_parents_mating = 2
no_of_weights = no_inputs*no_outputs + no_outputs
pop_size = (ind_per_pop, no_of_weights)

population = np.empty((ind_per_pop, no_inputs+1, no_outputs))
for _ in range(ind_per_pop):
    new_population = np.random.uniform(
        low=-1.0, high=1.0, size=(no_inputs+1, no_outputs))
    population[_] = new_population
# print(population)

fit = 0

for gen in range(no_generations):
    print("NewGeneration",gen)
    print("NewPop",population)
    pop_fitness = np.empty(ind_per_pop)
    for ind in range(ind_per_pop):

        # Reset entities after each individual training.
        pacman = Pacman(13, 19)
        pac_upd = 0
        blinky = Blinky(13, 11)
        inky = Inky(13, 7)
        inky.set_home((24, 3))
        pinky = Pinky(3, 22)
        pinky.set_home((2, 26))
        clyde = Clyde(23, 22)
        clyde.set_home((24, 26))
        entities = [pacman, blinky, inky, pinky, clyde]

        fitness = cal_ind_fitness(population[ind])
        fit = max(fitness,fit)
        pop_fitness[ind] = fitness
    print("Fitness", pop_fitness)
    parents = select_mating_pool(population, pop_fitness, num_parents_mating)
    # print("parents")
    # print(parents)
    offspring_crossover = crossover(parents,
                                    offspring_size=(ind_per_pop-num_parents_mating, no_inputs+1, no_outputs))
    print("Offsprings")
    print(offspring_crossover)

    offspring_mutation = mutation(offspring_crossover, num_mutations=10)
    print("Mutation")
    print(offspring_mutation)    

    population[0:num_parents_mating, :] = parents
    population[num_parents_mating:, :] = offspring_mutation

print(population)
print("MaxFitness",fit)

time.sleep(2)
pacman = Pacman(13, 19)
pac_upd = 0
blinky = Blinky(13, 11)
inky = Inky(13, 7)
inky.set_home((24, 3))
pinky = Pinky(3, 22)
pinky.set_home((2, 26))
clyde = Clyde(23, 22)
clyde.set_home((24, 26))
entities = [pacman, blinky, inky, pinky, clyde]

fitness = cal_ind_fitness_final(population[ind])
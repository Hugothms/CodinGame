import sys
import math
import copy
import time
import numpy as np
from enum import Enum
import random

circles=[range(0, 1), range(1, 7), range(7, 19), range(19, 37)]

costs_grow=[1, 3, 7]

best_cells_seed=[0, 1, 3, 5, 8, 10, 12, 14, 16, 18, 19, 22, 25, 28, 31, 34]

def get_initial_sun_exposed_cells(day:int):
    sun_exposed_cells = []
    for num in range(6, 13):
        sun_exposed_cells.append((num + 3 * day) % 18 + 19)
    return sun_exposed_cells

def print_debug(str, end='\n'):
    print(str, file=sys.stderr, flush=True, end=end)

def number_tree_size(trees:list, to_play:bool=True):
    nb_tree_level = [0, 0, 0, 0]
    for tree in trees:
        if (tree.is_mine == to_play):
            nb_tree_level[tree.size] += 1
    return nb_tree_level






class Cell:
    def __init__(self, cell_index, richness, neighbors):
        self.cell_index = cell_index
        self.richness = richness
        self.neighbors = neighbors

class Tree:
    def __init__(self, cell_index, size, is_mine, is_dormant):
        self.cell_index = cell_index
        self.size = size
        self.is_mine = is_mine
        self.is_dormant = is_dormant

class ActionType(Enum):
    WAIT = "WAIT"
    SEED = "SEED"
    GROW = "GROW"
    COMPLETE = "COMPLETE"

class Action:
    def __init__(self, type, target_cell_id=None, origin_cell_id=None):
        self.type = type
        self.target_cell_id = target_cell_id
        self.origin_cell_id = origin_cell_id

    def __str__(self):
        if self.type == ActionType.WAIT:
            return 'WAIT'
        elif self.type == ActionType.SEED:
            return f'SEED {self.origin_cell_id} {self.target_cell_id}'
        else:
            return f'{self.type.name} {self.target_cell_id}'

    @staticmethod
    def parse(action_string):
        split = action_string.split(' ')
        if split[0] == ActionType.WAIT.name:
            return Action(ActionType.WAIT)
        if split[0] == ActionType.SEED.name:
            return Action(ActionType.SEED, int(split[2]), int(split[1]))
        if split[0] == ActionType.GROW.name:
            return Action(ActionType.GROW, int(split[1]))
        if split[0] == ActionType.COMPLETE.name:
            return Action(ActionType.COMPLETE, int(split[1]))

    def sort_actions(self):
        return str(self)

class Game:
    def __init__(self):
        self.day = 0
        self.nutrients = 0
        self.board = []
        self.trees = []
        self.possible_actions = []
        self.sun = [0, 0]
        self.score = [0, 0]
        self.opponent_is_waiting = 0


    def get_trees_player(self, to_play:bool=True):
        return [trees for trees in self.trees if trees.is_mine == to_play]

    def get_seeds_player(self, to_play:bool=True):
        return [seeds for seeds in self.get_trees_player(to_play) if seeds.size == 0]

    def get_tree_at_index(self, start:int, end:int=None, to_play:bool=None):
        res_trees = []
        for tree in self.trees:
            if to_play is None or tree.is_mine == to_play:
                if end is None:
                    if tree.cell_id == start:
                        return tree
                else:
                    if start <= tree.cell_id <= end:
                        res_trees.append(tree)
        if end is None:
            return None
        return res_trees

    def get_cell_at_index(self, index:int):
        for cell in self.board:
            if cell.cell_id == index:
                return cell

    def position_is_optimal(self, action:Action):
        cell_action = self.get_cell_at_index(action.target_cell_id)
        for neighbor in cell_action.neighbors:
            if neighbor != -1 and self.get_cell_at_index(neighbor).neighbors[(i + 1) % 5] != -1:
                if self.get_cell_at_index(neighbor).neighbors[(i + 1) % 5] == action.target_cell_id:
                    return True
        return False

    def print_state_game(self):
        print_debug("day: " + str(self.day))

        print_debug("nutrients: " + str(self.nutrients))

        print_debug("sun: " + str(self.sun[0]))
        print_debug("score: " + str(self.score[1]))

        print_debug("opp_sun: " + str(self.sun[1]))
        print_debug("opp_score: " + str(self.score[1]))
        print_debug("opp_is_waiting: " + str(self.opponent_is_waiting))

        print_debug("number_of_trees: " + str(len(self.trees)))
        for tree in self.trees:
            print_debug(" -cell_id: " + str(tree.cell_id))
            print_debug("  size: " + str(tree.size))
            print_debug("  is_mine: " + str(tree.is_mine))
            print_debug("  is_dormant: " + str(tree.is_dormant))

        print_debug("number_of_possible_actions: " + str(len(self.possible_actions)))
        for possible_action in self.possible_actions:
            print_debug(" -possible_action: " + str(possible_action))









    def compute_next_action(self):
        """
        if self.score[1] > self.score[1] and not self.opponent_is_waiting and :
            #print_debug('Wait strategique ? pas sur mdr')
            return "WAIT Wait strategique ?"
        """
        smaller_cell = 37
        bigger_cell = 0
        best_action = Action(ActionType.WAIT)
        bigger_grow = -1
        output = 'WAIT'
        #self.possible_actions.sort(key=Action.sort_actions)

        # mode on plante a gogo
        if self.day < 7:
            for action in self.possible_actions:
                if (action.type == ActionType.GROW and action.target_cell_id in best_cells_seed):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        size_target = self.get_tree_at_index(action.target_cell_id).size
                        if (size_target > bigger_grow):
                            bigger_grow = size_target
                            best_action = action
                elif (action.type == ActionType.SEED and best_action.type != ActionType.GROW and action.target_cell_id in best_cells_seed):
                    if (action.target_cell_id > bigger_cell):
                        bigger_cell = action.target_cell_id
                        best_action = action
            if self.day == 0 and best_action.type == ActionType.WAIT:
                for action in self.possible_actions:
                    if action.type == ActionType.SEED:
                        best_action = action
        elif 6 <= self.day < 25:
            for action in self.possible_actions:
                #print_PA(action)
                # COMPLETE au centre if possible
                if (action.type == ActionType.COMPLETE and (len(self.get_trees_player()) > 8 or self.day > 21)):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        best_action = action
                # GROW en priorité a l'interieur
                elif (action.type == ActionType.GROW and best_action.type != ActionType.COMPLETE):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        size_target = self.get_tree_at_index(action.target_cell_id).size
                        if (size_target > bigger_grow):
                            bigger_grow = size_target
                            best_action = action
                # SEED if no GROW avalaible and day < 14
                elif (action.type == ActionType.SEED and best_action.type != ActionType.COMPLETE and best_action.type != ActionType.GROW and (self.day < 14 or len(self.get_trees_player())) == 1):
                    if (action.target_cell_id > bigger_cell):
                        bigger_cell = action.target_cell_id
                        best_action = action
        elif 15 <= self.day < 20:
            pass
        elif 20 <= self.day:
            pass
        return best_action

# END CLASS GAME











### GAME LOOP ###

number_of_cells = int(input())
game = Game()
for i in range(number_of_cells):
    cell_index, richness, neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5 = [int(j) for j in input().split()]
    game.board.append(Cell(cell_index, richness, [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5]))

while True:
    t0 = time.time()
    day = int(input())
    game.day = day
    nutrients = int(input())
    game.nutrients = nutrients
    sun, score = [int(i) for i in input().split()]
    game.sun[0] = sun
    game.score[1] = score
    opp_sun, opp_score, opp_is_waiting = [int(i) for i in input().split()]
    game.sun[1] = opp_sun
    game.score[1] = opp_score
    game.opponent_is_waiting = opp_is_waiting
    number_of_trees = int(input())
    game.trees.clear()
    for i in range(number_of_trees):
        inputs = input().split()
        cell_index = int(inputs[0])
        size = int(inputs[1])
        is_mine = inputs[2] != "0"
        is_dormant = inputs[3] != "0"
        game.trees.append(Tree(cell_index, size, is_mine, is_dormant))
    number_of_possible_actions = int(input())
    game.possible_actions.clear()
    for i in range(number_of_possible_actions):
        possible_action = input()
        game.possible_actions.append(Action.parse(possible_action))

    print(game.compute_next_action())

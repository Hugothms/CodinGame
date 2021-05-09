import sys
import math
from enum import Enum
import random

circles=[range(0, 1), range(1, 7), range(7, 19), range(19, 37)]
cost_grow=[1, 3, 7]


def print_debug(str):
    print(str, file=sys.stderr, flush=True)


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

def sort_actions(action):
    return str(action)

def print_PA(action):
    print("PA: " + action.type.name, file=sys.stderr, flush=True, end='')
    if (action.origin_cell_id != None):
        print(' ' + str(action.origin_cell_id), file=sys.stderr, flush=True, end='')
    if (action.target_cell_id != None):
        print(' ' + str(action.target_cell_id), file=sys.stderr, flush=True, end='')
    print('', file=sys.stderr, flush=True,)


class Game:
    def __init__(self):
        self.day = 0
        self.nutrients = 0
        self.board = []
        self.trees = []
        self.possible_actions = []
        self.my_sun = 0
        self.my_score = 0
        self.opponents_sun = 0
        self.opponent_score = 0
        self.opponent_is_waiting = 0



    def number_tree_lvl(self, lvl):
        #return [trees for trees in self.threes if trees.is_mine == 1]
        cpt = 0
        for tree in self.trees:
            print_debug("is_mine: " + tree.is_mine)
            if (tree.is_mine and tree.size == lvl):
                cpt += 1
        return cpt

    def get_my_trees(self):
        return [trees for trees in self.trees if trees.is_mine == 1]

    def get_tree_at_index(self, index):
        for tree in self.trees:
            if tree.cell_index == index:
                return tree

    def get_cell_at_index(self, index):
        for cell in self.board:
            if cell.cell_index == index:
                return cell



    def compute_next_action(self):
        """
        if self.my_score > self.opponent_score and not self.opponent_is_waiting and :
            #print_debug('Wait strategique ? pas sur mdr')
            return "WAIT Wait strategique ?"
        """
        """
        smaller_cell = 37
        bigger_cell = 0
        best_action = ActionType.WAIT
        bigger_grow = -1
        output = 'WAIT'
        #self.possible_actions.sort(key=sort_actions)
        for action in self.possible_actions:
            #print_PA(action)
            # COMPLETE if possible
            if (action.type == ActionType.COMPLETE and action.type.name <= best_action.name):
                best_action = action.type
                if (action.target_cell_id < smaller_cell):
                    smaller_cell = action.target_cell_id
                    output = action
            # GROW en priorité a l'interieur
            elif (action.type == ActionType.GROW and
            action.type.name <= best_action.name):
                best_action = action.type
                if (action.target_cell_id < smaller_cell):
                    smaller_cell = action.target_cell_id
                    size_target = self.get_tree_at_index(action.target_cell_id).size
                    if (size_target > bigger_grow):
                        bigger_grow = size_target
                        output = action
            # SEED if no GROW avalaible and day < 14
            elif (action.type == ActionType.SEED and
            action.type.name <= best_action.name and
            (self.day < 14 or len(self.get_my_trees())) == 1):
                best_action = action.type
                if (action.target_cell_id > bigger_cell):
                    bigger_cell = action.target_cell_id
                    output = action
        return output
        """
        return minimax(self, 10, -math.inf, math.inf, True)[1]





### MINIMAX ALGORITHM ###

def apply_action(game, action, action_opp): # OK
    if action.type == ActionType.WAIT:
        return
    elif action.type == ActionType.SEED:
        if action_opp.type == ActionType.SEED and action.target_cell_id == action_opp.target_cell_id:
            print_debug("you and opponenet tried to SEED on the same cell -> ABORT")
            return
        game.my_sun -= len([tree for tree in game.threes if tree.is_mine == 1 and tree.size == 0])
        game.trees.append(Tree(action.target_cell_id, 0, 1, 1))
        game.get_tree_at_index(action.origin_cell_id).is_dormant = True
        return
    elif action.type == ActionType.GROW:
        tree_to_grow = game.get_tree_at_index(action.target_cell_id)
        game.my_sun -= cost_grow[tree_to_grow.size] + len([tree for tree in game.threes if tree.is_mine == 1 and tree.size == tree_to_grow.size + 1])
        tree_to_grow.size += 1
    elif action.type == ActionType.COMPLETE:
        game.trees.reaction(game.get_tree_at_index(action.target_cell_id))
        game.my_sun -= 4
        game.my_score += game.nutrients + (game.get_cell_at_index(action.target_cell_id).richness - 1) * 2
        game.nutrients -= 1
    else:
        print_debug("error action invalid")
    return game


def all_seed_actions_from_tree(game, cell_index, depth, visited_cells_ids): # OK
    actions = []
    for neighbor_id in game.get_cell_at_index(cell_index).neighbors:
        if neighbor_id not in visited_cells_ids:
            visited_cells_ids.append(neighbor_id)
            actions.append(Action(ActionType.SEED, neighbor_id, cell_index))
            if depth > 0:
                actions.extend(all_seed_actions_from_tree(game, neighbor_id, depth - 1, visited_cells_ids))
    return actions

def find_all_possible_actions(game): # OK
    actions = []
    for tree in game.trees:
        if not tree.is_dormant:
            if tree.size == 3:
                actions.append(Action(ActionType.COMPLETE, tree.cell_index))
            else:
                actions.append(Action(ActionType.GROW, tree.cell_index))
            if (tree.size > 0):
                visited_cells_ids = [tree.cell_index]
                actions.extend(all_seed_actions_from_tree(game, tree.cell_index, tree.size, visited_cells_ids))
    # for external_cell in range(19, 37):
    #     actions.append(Action(ActionType.SEED, external_cell))
    return actions.sort(key=sort_actions) # to try best cadidate first (like COMPLETE action before others)


# if worse than options already explored -> do not explore
def minimax(game, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or game.day == 25: # or time is up
        return (game.my_score - game.opp_score, None) # static evaluation of game (todo: to enhance)
    if maximizingPlayer:
        maxEval = -math.inf
        for action in find_all_possible_actions(game):
            print_debug(str(action))
            game = apply_action(game, action, action_opp)
            eval = minimax(child, depth - 1, alpha, beta, false)[0]
            if eval > maxEval:
                maxEval = eval
                best_action = action
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return (maxEval, best_action)
    else:
        minEval = math.inf
        for action in find_all_possible_actions(game):
            game = apply_action(game, action, action_opp)
            eval = minimax(child, depth - 1, alpha, beta, true)[0]
            if eval < minEval:
                minEval = eval
                best_action = action
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return (minEval, best_action)




number_of_cells = int(input())
game = Game()
for i in range(number_of_cells):
    cell_index, richness, neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5 = [int(j) for j in input().split()]
    print("cell_index: " + str(cell_index), file=sys.stderr, flush=True)
    print("richness: " + str(richness), file=sys.stderr, flush=True)
    print("neigh_0: " + str(neigh_0), file=sys.stderr, flush=True)
    print("neigh_1: " + str(neigh_1), file=sys.stderr, flush=True)
    print("neigh_2: " + str(neigh_2), file=sys.stderr, flush=True)
    print("neigh_3: " + str(neigh_3), file=sys.stderr, flush=True)
    print("neigh_4: " + str(neigh_4), file=sys.stderr, flush=True)
    print("neigh_5: " + str(neigh_5), file=sys.stderr, flush=True)
    game.board.append(Cell(cell_index, richness, [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5]))


while True:
    _day = int(input())
    print("_day: " + str(_day), file=sys.stderr, flush=True)
    game.day = _day

    nutrients = int(input())
    print("nutrients: " + str(nutrients), file=sys.stderr, flush=True)
    game.nutrients = nutrients

    sun, score = [int(i) for i in input().split()]
    print("sun: " + str(sun), file=sys.stderr, flush=True)
    print("score: " + str(score), file=sys.stderr, flush=True)
    game.my_sun = sun
    game.my_score = score

    opp_sun, opp_score, opp_is_waiting = [int(i) for i in input().split()]
    print("opp_sun: " + str(opp_sun), file=sys.stderr, flush=True)
    print("opp_score: " + str(opp_score), file=sys.stderr, flush=True)
    print("opp_is_waiting: " + str(opp_is_waiting), file=sys.stderr, flush=True)
    game.opponent_sun = opp_sun
    game.opponent_score = opp_score
    game.opponent_is_waiting = opp_is_waiting

    number_of_trees = int(input())
    print("number_of_trees: " + str(number_of_trees), file=sys.stderr, flush=True)
    game.trees.clear()
    for i in range(number_of_trees):
        inputs = input().split()

        cell_index = int(inputs[0])
        print(" -cell_index: " + str(cell_index), file=sys.stderr, flush=True)

        size = int(inputs[1])
        print("  size: " + str(size), file=sys.stderr, flush=True)

        is_mine = inputs[2] != "0"
        print("  is_mine: " + str(is_mine), file=sys.stderr, flush=True)

        is_dormant = inputs[3] != "0"
        print("  is_dormant: " + str(is_dormant), file=sys.stderr, flush=True)

        game.trees.append(Tree(cell_index, size, is_mine == 1, is_dormant))

    number_of_possible_actions = int(input())
    print("number_of_possible_actions: " + str(number_of_possible_actions), file=sys.stderr, flush=True)

    game.possible_actions.clear()
    for i in range(number_of_possible_actions):
        possible_action = input()
        print(" -possible_action: " + str(possible_action), file=sys.stderr, flush=True)
        game.possible_actions.append(Action.parse(possible_action))

    print(game.compute_next_action())

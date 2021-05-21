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

def get_initial_sun_exposed_cells(day):
    sun_exposed_cells = []
    for num in range(6, 13):
        sun_exposed_cells.append((num + 3 * day) % 18 + 19)
    return sun_exposed_cells

def print_debug(str, end='\n'):
    print(str, file=sys.stderr, flush=True, end=end)








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

# def print_PA(action):
#     print("PA: " + action.type, file=sys.stderr, flush=True, end='')
#     if (action.origin_cell_id is not None):
#         print(' ' + str(action.origin_cell_id), file=sys.stderr, flush=True, end='')
#     if (action.target_cell_id is not None):
#         print(' ' + str(action.target_cell_id), file=sys.stderr, flush=True, end='')
#     print('', file=sys.stderr, flush=True,)


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

    def number_tree_lvl(self, lvl, to_play=True):
        cpt = 0
        for tree in self.trees:
            if (tree.is_mine == to_play and tree.size == lvl):
                cpt += 1
        return cpt

    def get_trees_player(self, to_play=True):
        return [trees for trees in self.trees if trees.is_mine == to_play]

    def get_seeds_player(self, to_play=True):
        return [seeds for seeds in self.get_trees_player(to_play) if seeds.size == 0]

    def get_tree_at_index(self, index):
        for tree in self.trees:
            if tree.cell_index == index:
                return tree

    def get_cell_at_index(self, index):
        for cell in self.board:
            if cell.cell_index == index:
                return cell

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
            print_debug(" -cell_index: " + str(tree.cell_index))
            print_debug("  size: " + str(tree.size))
            print_debug("  is_mine: " + str(tree.is_mine))
            print_debug("  is_dormant: " + str(tree.is_dormant))

        print_debug("number_of_possible_actions: " + str(len(self.possible_actions)))
        for possible_action in self.possible_actions:
            print_debug(" -possible_action: " + str(possible_action))














    ### SIMULATION ###

    def apply_actions(self, action, action_opp, to_play=True): # OK
        if action.type == ActionType.WAIT:
            return self
        elif action.type == ActionType.SEED:
            if action_opp.type == ActionType.SEED and action.target_cell_id == action_opp.target_cell_id:
                # print_debug("you and opponenet tried to SEED on the same cell -> ABORT")
                return self
            # print_debug("pos of seeds:")
            # for tree in [tree for tree in self.get_trees_player() if (tree.size == 0)]:
                # print_debug(tree.cell_index) # issue on calcul
            # print_debug("cost of SEED: " + str(len([tree for tree in self.get_trees_player() if (tree.size == 0)])))
            self.sun[to_play == True] -= len([tree for tree in self.get_trees_player(to_play) if (tree.size == 0)])
            self.trees.append(Tree(action.target_cell_id, 0, 1, 1))
            self.get_tree_at_index(action.origin_cell_id).is_dormant = True
            return self
        elif action.type == ActionType.GROW:
            tree_to_grow = self.get_tree_at_index(action.target_cell_id)
            # print_debug(costs_grow[tree_to_grow.size] + len([tree for tree in self.get_trees_player() if tree.size == (tree_to_grow.size + 1)]))
            self.sun[to_play == True] -= costs_grow[tree_to_grow.size] + len([tree for tree in self.get_trees_player(to_play) if tree.size == (tree_to_grow.size + 1)])
            tree_to_grow.size += 1
        elif action.type == ActionType.COMPLETE:
            self.trees.remove(self.get_tree_at_index(action.target_cell_id))
            self.sun[to_play == True] -= 4
            self.score[1] += self.nutrients + (self.get_cell_at_index(action.target_cell_id).richness - 1) * 2
            self.nutrients -= 1
        # else:
            # print_debug("error action invalid")
        return self



    # should be replaced to find only interesting seed actions (with position_is_optimal)
    def all_seed_actions_from_tree(self, cell_index, depth, visited_cells_ids): # OK
        actions = []
        neighbors = self.get_cell_at_index(cell_index).neighbors
        for neighbor_id in neighbors:
            if neighbor_id not in visited_cells_ids and self.get_cell_at_index(neighbor_id).richness > 0 and self.get_tree_at_index(neighbor_id) is None:
                visited_cells_ids.append(neighbor_id)
                actions.append(Action(ActionType.SEED, neighbor_id, cell_index))
                if depth > 0:
                    actions.extend(self.all_seed_actions_from_tree(cell_index, depth - 1, visited_cells_ids))
        # for action in actions:
        #     print_debug(action)
        return actions

    def find_all_possible_actions(self, to_play=True): # OK
        actions = [Action(ActionType.WAIT)]
        player_trees = self.get_trees_player(to_play)
        for tree in player_trees:
            if not tree.is_dormant:
                if self.sun[to_play == True] >= 4 and tree.size == 3:
                    actions.append(Action(ActionType.COMPLETE, tree.cell_index))
                elif tree.size < 3 and self.sun[to_play == True] >= costs_grow[tree.size] + len([trees for trees in player_trees if trees.size == tree.size]):
                    actions.append(Action(ActionType.GROW, tree.cell_index))
                if (tree.size > 0) and self.sun[to_play == True] >= len(self.get_seeds_player(to_play)):
                    visited_cells_ids = [-1, tree.cell_index]
                    actions.extend(self.all_seed_actions_from_tree(tree.cell_index, tree.size, visited_cells_ids))
        # print_debug("***********ACTIONS:***********")
        # actions.sort(key=Action.sort_actions)
        # for action in actions:
        #     print_debug(action)
        # print_debug("**************")
        return actions # to try best cadidate first (like COMPLETE action before others)

    def position_is_optimal(self, action):
        cell_action = self.get_cell_at_index(action.target_cell_id)
        for neighbor in cell_action.neighbors:
            if neighbor != -1 and self.get_cell_at_index(neighbor).neighbors[(i + 1) % 5] != -1:
                if self.get_cell_at_index(neighbor).neighbors[(i + 1) % 5] == action.target_cell_id:
                    return True;
        return False;

    def interesting_seed_actions_from_tree(self, cell_index, depth, visited_cells_ids): # OK
        actions = []
        neighbors = self.get_cell_at_index(cell_index).neighbors
        for neighbor_id in neighbors:
            if neighbor_id not in visited_cells_ids and self.get_cell_at_index(neighbor_id).richness > 0 and self.get_tree_at_index(neighbor_id) is None:
                visited_cells_ids.append(neighbor_id)
                seed_action = Action(ActionType.SEED, neighbor_id, cell_index)
                if self.position_is_optimal(seed_action):
                    actions.append(seed_action)
                if depth > 0:
                    actions.extend(self.interesting_seed_actions_from_tree(cell_index, depth - 1, visited_cells_ids))
        return actions

    def find_interesting_actions(self, to_play=True):
        actions = [Action(ActionType.WAIT)]
        player_trees = self.get_trees_player(to_play)
        player_nb_seeds = len(self.get_seeds_player(to_play))
        for tree in player_trees:
            if not tree.is_dormant:
                if tree.size == 3:
                    if self.sun[to_play == True] >= 4:
                        actions.append(Action(ActionType.COMPLETE, tree.cell_index))
                elif self.sun[to_play == True] >= costs_grow[tree.size] + len([trees for trees in player_trees if trees.size == tree.size]):
                    actions.append(Action(ActionType.GROW, tree.cell_index))
                if tree.size > 1 and self.sun[to_play == True] >= player_nb_seeds:
                    visited_cells_ids = [-1, tree.cell_index]
                    actions.extend(self.interesting_seed_actions_from_tree(tree.cell_index, tree.size, visited_cells_ids))
                    # actions.extend(self.all_seed_actions_from_tree(tree.cell_index, tree.size, visited_cells_ids))
        # print_debug("***********ACTIONS:***********")
        # actions.sort(key=sort_actions)
        # for action in actions:
        #     print_debug(action)
        return actions # to try best cadidate first (like COMPLETE action before others)

    # def filter_action(self, action):
    #     # return True
    #     if action.type == ActionType.WAIT:
    #         return True
    #     elif action.type == ActionType.SEED:
    #         return action.target_cell_id not in self.get_cell_at_index(action.target_cell_id).neighbors
    #     elif action.type == ActionType.GROW:
    #         return True
    #     elif action.type == ActionType.COMPLETE:
    #         return True
    #     return False

    # def filter_actions(self, actions):
    #     filtered = []
    #     for action in actions:
    #         if filter_action(self, action):
    #             filtered.append(action)
    #     return filtered

    def sun_in_row(self, sun_exposed_cell, shadow_range, shadow_size_tree, to_play=True):
        # print_debug("start sun_in_row: " + str(sun_exposed_cell) + '\t' + str(shadow_range))
        if sun_exposed_cell == -1:
            return (0, 0)
        sun = [0, 0]
        tree = self.get_tree_at_index(sun_exposed_cell)
        if tree is not None:
            # print_debug('tree')
            if shadow_range == 0 or tree.size > shadow_size_tree:
                if tree.is_mine:
                    sun[to_play == True] += tree.size
                else:
                    sun[1] += tree.size
            shadow_size_tree = tree.size
            shadow_range = max(tree.size + 1, shadow_range) #OK
        suns = self.sun_in_row(self.get_cell_at_index(sun_exposed_cell).neighbors[day % 6], max(shadow_range - 1, 0), shadow_size_tree)
        sun[to_play == True] += suns[0]
        sun[1] += suns[1]
        return (sun[to_play == True], sun[1])

    def new_turn(self, new_day=False, to_play=True):
        initial_sun_exposed_cells = get_initial_sun_exposed_cells(day)
        for sun_exposed_cell in initial_sun_exposed_cells:
            # print_debug("+++++++++++++++++++")
            suns = self.sun_in_row(sun_exposed_cell, 0, 0)
            self.sun[to_play == True] += suns[0]
            self.sun[1] += suns[1]
            # print_debug("MINE: " + str(suns[0]))
            # print_debug("OPPO: " + str(suns[1]))
        # print_debug("final sun[to_play == True]: " + str(self.sun[to_play == True]))
        # print_debug("final sun[1]: " + str(self.sun[1]))
        if new_day:
            self.day += 1
            for tree in self.trees:
                if tree.is_dormant:
                    tree.is_dormant = False
        return self

    def evalutation_score_position(self, to_play=True):
        score = (self.score[to_play == True] - self.score[to_play == False])
        suns = (self.sun[to_play == True] - self.score[to_play == False]) / 4
        trees = self.get_trees_player()
        score_tree = 0
        if self.day < 0:
            trees_sizes = [0, 0, 0, 0]
            for tree in trees:
                trees_sizes[tree.size] += 1
            for i in range(4):
                score_tree += (i + 1) * trees_sizes[i]
        # print_debug('my_score:   ' + str(self.score[to_play == True]), '\t')
        # print_debug('score:      ' + str(score))
        # print_debug('my_sun:     ' + str(self.sun[to_play == True]), '\t')
        # print_debug('sun:        ' + str(sun))
        # print_debug('score_tree: ' + str(score_tree))
        # print_debug('')
        return score + suns + score_tree

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
        if self.day < 6:
            for action in self.possible_actions:
                if (action.type == ActionType.GROW and
                action.target_cell_id in best_cells_seed):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        size_target = self.get_tree_at_index(action.target_cell_id).size
                        if (size_target > bigger_grow):
                            bigger_grow = size_target
                            best_action = action
                elif (action.type == ActionType.SEED and
                best_action.type != ActionType.GROW and
                action.target_cell_id in best_cells_seed):
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
                if (action.type == ActionType.COMPLETE and
                (len(self.get_trees_player()) > 8 or self.day > 21)):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        best_action = action
                # GROW en priorité a l'interieur
                elif (action.type == ActionType.GROW and
                best_action.type != ActionType.COMPLETE):
                    if (action.target_cell_id < smaller_cell):
                        smaller_cell = action.target_cell_id
                        size_target = self.get_tree_at_index(action.target_cell_id).size
                        if (size_target > bigger_grow):
                            bigger_grow = size_target
                            best_action = action
                # SEED if no GROW avalaible and day < 14
                elif (action.type == ActionType.SEED and
                best_action.type != ActionType.COMPLETE and
                best_action.type != ActionType.GROW and
                (self.day < 14 or len(self.get_trees_player())) == 1):
                    if (action.target_cell_id > bigger_cell):
                        bigger_cell = action.target_cell_id
                        best_action = action
        elif 15 <= self.day < 20:
            pass
        elif 20 <= self.day:
            pass
        return best_action

# END CLASS GAME
























def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, model, state, to_play):

        root = Node(0, to_play)

        # EXPAND root
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, to_play=True, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, to_play=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, to_play=True)
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1





# class Node:
#     def __init__(self, win=0, nb_visit=0):
#         self.win = win
#         self.nb_visit = nb_visit

# # main function for the Monte Carlo Tree Search
# def monte_carlo_tree_search(game, t0):
#     node = Node()
#     while time.time() - t0 < 90:
#         leaf = traverse(game)
#         simulation_result = rollout(leaf)
#         backpropagate(leaf, simulation_result)
#     return best_child(game)

# def best_uct(node):
#     return node.win / node.nb_visit + 2 * math.sqrt(math.log(pere_expl) / node.nb_visit)

# # function for node traversal
# def traverse(game, node):
#     for action in find_all_possible_actions(game):
#         node = Node()
#         game = best_uct(game)
#     # in case no children are present / node is terminal
#     return pick_univisted(node.children) or node


# # function for the result of the simulation
# def rollout(node):
#     while non_terminal(node):
#         node = rollout_policy(node)
#     return result(node)


# # function for randomly selecting a child node
# def rollout_policy(node):
#     return pick_random(node.children)


# # function for backpropagation
# def backpropagate(node, result):
#     if is_root(node):
#         return
#     node.stats = update_stats(node, result)
#     backpropagate(node.parent)


# # function for selecting the best child
# # node with highest number of visits
# def best_child(node):
#     pick child with highest number of visits












### MINIMAX ALGORITHM ###

# if worse than options already explored --> do not explore
def minimax(game, depth, alpha=-math.inf, beta=math.inf, best_actions=[], time_init=time.time(), time_max=0.009):
    global seen
    global printed
    if depth not in seen:
        seen.append(depth)
    elif depth not in printed:
        printed.append(depth)
        print_debug(str(depth) + '\t' + str(time.time() - time_init))

    global position_count
    position_count += 1
    # print_debug("depth: " + str(depth))
    if depth == 0 or game.day == 25 or time.time() - time_init > time_max: # or time is up
        eval = game.evalutation_score_position()
        # print_debug("eval: " + str(eval) + '\n')
        return (eval, None, []) # static evaluation of game (todo: to enhance)
    max_eval = -math.inf
    for action in game.find_interesting_actions(True):
    #     for action_opp in game.find_interesting_actions(False):
    # for action in game.find_all_possible_actions(True):
        # for action_opp in game.find_all_possible_actions(False):
        action_opp = Action(ActionType.WAIT)
        # print_debug("Turn actions: " + str(action) + ' /// ' + str(action_opp))
        next_game = copy.deepcopy(game)
        next_game.apply_actions(action, action_opp, True)
        # next_game.apply_actions(action_opp, action, False)
        next_game = next_game.new_turn(action.type == ActionType.WAIT and action_opp.type == ActionType.WAIT)
        res = minimax(next_game, depth - 1, alpha, beta, best_actions, time_init, time_max)
        eval = res[0]
        if eval > max_eval:
            max_eval = eval
            best_action = action
            best_actions = res[2]
        # alpha = max(alpha, eval)
        # if beta <= alpha:
        #     break
    best_actions.append(best_action)
    return (max_eval, best_action, best_actions)



### GAME LOOP ###

number_of_cells = int(input())
game = Game()
for i in range(number_of_cells):
    cell_index, richness, neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5 = [int(j) for j in input().split()]
    # print_debug("cell_index: " + str(cell_index))
    # print_debug("richness: " + str(richness))
    # print_debug("neigh_0: " + str(neigh_0))
    # print_debug("neigh_1: " + str(neigh_1))
    # print_debug("neigh_2: " + str(neigh_2))
    # print_debug("neigh_3: " + str(neigh_3))
    # print_debug("neigh_4: " + str(neigh_4))
    # print_debug("neigh_5: " + str(neigh_5))
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

    # game.print_state_game()
    # for action in game.possible_actions:
    #     print_debug('@@@@@@@@@@@@@@@@@@@')
    #     print_debug(action)
    # print(game.compute_next_action())
    # for action in game.find_all_possible_actions(True):
    #     print_debug(action)

    global position_count
    position_count = 0
    global seen
    seen = []
    global printed
    printed = []
    res = minimax(game, depth=8, time_max=1000)
    print(res[1])
    print_debug("=======> Score: " + str(res[0]))
    print_debug("*Evaluated Pos: " + str(position_count))
    for action in res[2]:
        print_debug(action)

    # mcts = MCTS(game,)
    # print(minimax(game, 8))

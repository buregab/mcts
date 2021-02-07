import logging
import numpy as np
import pyspiel

from copy import deepcopy

# the exploration parameter in UCB
C = np.sqrt(2)

class Node:
    def __init__(self, state, player=0, parent=None, action=None):
        self.set_state(state)
        self.player = player
        self.rollouts = 0
        self.wins = 0
        self.children = []
        self.parent = parent
        self.action = action
        self.max_children = len(self._state.legal_actions())

    def ucb1(self, N):
        if self.rollouts == 0:
            return C * np.sqrt(np.log(N))
        return self.wins / self.rollouts + C * np.sqrt(np.log(N) / self.rollouts)

    def get_best_child(self):
        ucb_vals = np.zeros(len(self.children))
        for i in range(len(self.children)):
            ucb_vals[i] = self.children[i].ucb1(self.rollouts)
        selected_index = np.argmax(ucb_vals)
        logging.info("state") 
        logging.info(str(self._state))
        logging.info("ucb vals")
        logging.info(" ".join([str(e) for e in ucb_vals]))
        return selected_index

    def update_node(self, returns):
        self.rollouts += 1 
        if returns[self.player] == 1.0:
            self.wins += 1.0
        elif returns[self.player] == 0.0:
            self.wins += 0.5 

    def set_state(self, state):
        self._state = deepcopy(state)
    
    def get_state(self):
        return deepcopy(self._state)

    def is_action_tried(self, action):
        for child in self.children:
            if child.action == action:
                return True
        return False
    
class MCTS:
    def __init__(self, state):
        self.root = Node(state)
        self.state_to_node = {str(self.root.get_state()): self.root}

    def selection(self):
        self._selection(self.root)

    # a helper function to recursively traverse the game tree
    def _selection(self, root):
        # base case: expand a new child node
        if len(root.children) != root.max_children:
            self.expansion(root)
            return root
        elif root.get_state().is_terminal():
            self.expansion(root)
            return root

        # recursive case: traverse the tree according to ucb values
        selected_index = root.get_best_child()
        returns = self._selection(root.children[selected_index])
        return root.children[selected_index]

    # randomly create a new game node from a leaf node and perform a simulation
    def expansion(self, node):
        state = node.get_state()
        if state.is_terminal():
            returns = state.returns()
            self.backpropagation(node.parent, returns)
            return
        else:
            # pick an action which has not previously been tried
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            # the below is wrong because we can arrive at the same state different ways
            while(node.is_action_tried(action)):
                state = node.get_state()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
            child_node = Node(state, player=state.current_player(), parent=node, action=action)
            self.state_to_node[str(state)] = child_node
            node.children.append(child_node)
            returns = self.simulation(child_node.get_state())

            if not child_node.get_state().is_terminal():
                self.backpropagation(child_node, returns)
            else:
                self.backpropagation(node, returns)
        return

    # recursively simulate starting from a given state 
    # the returns are alwasy in the format [player_0_return, player_1_return]
    def simulation(self, state):
        if state.is_terminal():
            return state.returns()

        action = np.random.choice(state.legal_actions())
        state.apply_action(action)
        return self.simulation(state)

    def backpropagation(self, node, returns):
        while(node):
            node.update_node(returns)
            node = node.parent
        return

if __name__ == "__main__":
    mcts = MCTS()
    num_rollouts = 1000
    for i in range(num_rollouts):
        print(i)
        mcts.selection()
    print("finished!")

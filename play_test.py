import getopt
import logging
import pyspiel
import sys

from mcts import MCTS, bfs, dfs
from numpy.core.numeric import full

logging.basicConfig(filename='mcts.log', level=logging.DEBUG)

def full_game(agent_player=0):
    num_rollouts = 1000
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    while(not state.is_terminal()):
        if state.current_player() == agent_player:
            mcts = MCTS(state, player=agent_player)
            for i in range(num_rollouts):
                mcts.selection()
            child_index = mcts.root.get_best_child(C=0)
            action = mcts.root.children[child_index].action
        else:
            print(state)
            legal_actions = state.legal_actions()
            print("Your legal actions are:", legal_actions)
            valid_input = False
            action = None
            while(not valid_input):
                raw_input = input("Select a cell.")
                try:
                    action = int(raw_input)
                    assert action in legal_actions
                    valid_input = True
                except:
                    print("Invalid input.")
        state.apply_action(action)

    print(state)
    if state.returns()[agent_player] == 1.0:
        print("You lost.")
    elif state.returns()[agent_player] == 0.0:
        print("You tied.")
    else:
        print("You won")

def test_last_step():
    game = pyspiel.load_game("tic_tac_toe")
    state = game.deserialize_state('5\n0\n7\n6\n2\n8\n4\n1\n')
    mcts = MCTS(state)
    num_rollouts = 10
    for i in range(num_rollouts):
        mcts.selection()

def test_first_step():
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    mcts = MCTS(state, player=0)
    num_rollouts = 1000
    for i in range(num_rollouts):
        mcts.selection()
    bfs(mcts.root)

def process_args(argv):
    try:
        opts, args = getopt.getopt(argv, "p:", ["player="])
    except getopt.GetoptError:
        print("argument error")
        sys.exit()

    agent_player = 0    
    for opt, arg in opts:
        if opt in ("-p", "--player"):
            player = int(arg)
            agent_player = player ^ 1
    full_game(agent_player=agent_player)

if __name__ == "__main__":
    process_args(sys.argv[1:])
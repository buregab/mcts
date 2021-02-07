import logging
import pyspiel

from mcts import MCTS

logging.basicConfig(filename='mcts.log', level=logging.INFO)
num_rollouts = 5000
game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
while(not state.is_terminal()):
    if state.current_player() == 0:
        mcts = MCTS(state)
        for i in range(num_rollouts):
            mcts.selection()
        child_index = mcts.root.get_best_child()
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
if state.returns()[0] == 1.0:
    print("You lost.")
elif state.returns()[0] == 0.0:
    print("You tied.")
else:
    print("You won")

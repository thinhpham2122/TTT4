from TTT4 import TTT4
import numpy as np
from tensorflow.keras.models import load_model


def get_state(board_l, player):
    if player == 1:
        state_l = board_l[:]
        state_l.append(-1)
    else:
        state_l = board_l[:]
        state_l.append(1)
    return [state_l]


name = '100.14'
model = load_model(f'keras_model/{name}')
board = TTT4()
end = False
while not end:
    player_turn = board.player
    if player_turn == 1:
        state = get_state(board.board[:], int(player_turn))
        action = np.argmax(model.predict(np.array(state)))
        ret = board.play(action)
        if 'invalid' in ret:
            print('AI lose')
            exit()
    else:
        board.print_board()
        ret = board.play(int(input('Human turn:')))
        print(ret)

    if 'win' in ret or 'draw' in ret:
        print(ret)
        board.print_board()
        end = True

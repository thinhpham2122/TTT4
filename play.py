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


name = 'AI_model'
first = 1  # 1:ai first, 2:human first
model = load_model(f'keras_model/{name}')
board = TTT4()
end = False
while not end:
    player_turn = board.player
    if player_turn == first:
        state = get_state(board.board[:], int(player_turn))
        action = np.argmax(model.predict(np.array(state)))
        ret = board.play(action)
        if 'invalid' in ret:
            print('Invalid move. AI lose')
            exit()
        if 'win' in ret:
            print('Ai win')
        if 'draw' in ret:
            print('draw')
    else:
        board.print_board()
        ret = board.play(int(input('Human turn:')))
        if 'invalid' in ret:
            print('Invalid move. Try again')
        if 'win' in ret:
            print('Human win')
        if 'draw' in ret:
            print('draw')
    if 'win' in ret or 'draw' in ret:
        end = True
'''
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
'''

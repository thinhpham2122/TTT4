from TTT4 import TTT4
from agent import Agent
import numpy as np
# import tensorflow.keras.backend as K
# K.clear_session()


def get_state(board, player):
    if player == 1:
        state = board[:]
        state.append(-1)
    else:
        state = board[:]
        state.append(1)
    return [state]


def get_next_state(board, player, ai):
    board_l = TTT4()
    board_l.board = board
    state = get_state(board, player)
    ret = board_l.play(ai.act(state))
    if 'invalid' in ret:
        i = 0
        while 'invalid' in ret and i < 16:
            ret = board_l.play(i)
            i += 1
    if 'invalid' in ret:
        return None
    next_board = board_l.board
    next_player = board_l.player
    return get_state(next_board, next_player)


def get_reward(ret):
    reward = 0
    if 'win' in ret:
        reward = 1000
    elif 'invalid' in ret:
        reward = -1000
    return reward


def main():
    student = Agent(17, 16)
    game_n = 0
    while True:
        board = TTT4()
        end = False
        game_n += 1
        while not end:
            player_turn = board.player
            state = get_state(board.board[:], int(player_turn))
            action = student.act(np.array(state))
            ret = board.play(action)
            if 'invalid' in ret:
                i = 0
                while 'invalid' in ret and i < 16:
                    ret = board.play(i)
                    i += 1
            reward = get_reward(ret)
            next_state = get_next_state(board.board[:], int(board.player), student)
            if next_state:
                student.memory.append([state, action, reward, next_state][:])

            print(f'{game_n}: {board.played}/16 {action} {ret}: player {player_turn}')
            if 'win' in ret or 'draw' in ret:
                board.print_board()
                end = True
        student.exp_replay()

        student.model.save(f'keras_model/game{game_n}')


if __name__ == '__main__':
    main()

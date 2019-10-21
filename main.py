from TTT4 import TTT4
from agent import Agent
import numpy as np
from collections import deque


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
    board_l.player = player
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


def run(trial, memory):
    try:
        student = Agent(17, 16, model_name=name)
    except:
        student = Agent(17, 16)
    student.memory = memory if memory else student.memory
    game_n = 0
    for _ in range(trial):
        board = TTT4()
        end = False
        game_n += 1
        while not end:
            player_turn = board.player
            state = get_state(board.board[:], int(player_turn))
            action = student.act(np.array(state))
            ret = board.play(action)
            reward = get_reward(ret)
            if 'invalid' in ret:
                student.memory.append([state, action, reward, state][:])
                student.exp_replay()
                continue
            next_state = get_next_state(board.board[:], int(board.player), student)
            if next_state:
                student.memory.append([state, action, reward, next_state][:])

            print(f'{game_n}: {board.played}/16 {action} {ret}: player {player_turn-1}')
            board.print_board()

            if 'win' in ret or 'draw' in ret:
                end = True
        student.exp_replay()
        student.model.save(f'keras_model/{name}.h5')
    return student.memory


name = 'ai1'
try:
    mem = deque(maxlen=1000)
    for i in np.load('mem.npy', allow_pickle=True):
        mem.append(i)
except:
    mem = deque(maxlen=1000)

print(len(mem))
mem = run(1, mem)
np.save('mem', mem)


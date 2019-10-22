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
    ret = board_l.play(np.argmax(ai.predict(state)))
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
        reward = 100000000
    elif 'invalid' in ret:
        reward = -1
    return reward


def get_events(state, board, player, ai):
    events_l = []
    for i in range(len(board)):
        new_board = TTT4()
        new_board.board = board[:]
        new_board.player = int(player)
        ret = new_board.play(i)
        reward = get_reward(ret)
        if 'invalid' in ret or 'win' in ret:
            events_l.append([state, i, reward, None][:])
            continue

        next_state = get_next_state(new_board.board[:], int(new_board.player), ai)
        events_l.append([state, i, reward, next_state][:])
    return events_l


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
        student.epsilon_min = (len(student.memory)-1500)/1500
        while not end:
            player_turn = int(board.player)
            current_board = board.board[:]
            state = get_state(current_board, player_turn)
            action = student.act(np.array(state))
            ret = board.play(action)
            events = get_events(state, current_board, player_turn, student.model) if 'invalid' in ret else []

            for event in events:
                student.memory.append(event)
                if 'win' in ret:
                    student.memory[-17][2] = -1 if event[2] != 100000000 else 500
                    student.memory[-17][3] = None
                    print('xxxxx', student.memory[-17])

            if 'invalid' in ret:
                break

            print(f'{game_n}: {board.played}/16 {action} {ret}: player {player_turn-1}')
            board.print_board()

            if 'win' in ret or 'draw' in ret:
                end = True
    student.exp_replay()
    student.model.save(f'keras_model/{name}.h5')
    return student.memory


name = 'ai1'
try:
    mem = deque(maxlen=3000)
    for i in np.load('mem.npy', allow_pickle=True):
        mem.append(i)
except:
    mem = deque(maxlen=3000)

print(len(mem))

mem = run(10, mem)
np.save('mem', mem)


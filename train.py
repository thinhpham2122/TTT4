from TTT4 import TTT4
from agent import Agent
import numpy as np
import random
from collections import deque


def get_state(board, player):
    if player == 1:
        state = board[:]
        state.append(-1)
    else:
        state = board[:]
        state.append(1)
    return np.array([state])


def get_next_state(board, player, ai):
    board_l = TTT4()
    board_l.board = board
    board_l.player = player
    state = get_state(board, player)
    ret = board_l.play(np.argmax(ai.predict(state)))
    if 'invalid' in ret:
        empty_index = []
	for i in range(len(board)):
            if not board[i]:
                empty_index.append(i)
        ret = board_l.play(empty_index[random.randrange(len(empty_index)))
    next_board = board_l.board
    next_player = board_l.player
    return get_state(next_board, next_player), ret


def get_reward(ret, next_ret):
    if 'win' in ret:
        return 1
    elif 'invalid' in ret or 'win' in next_ret:
        return -1
    else:
        return 0


def get_events(state, board, player, ai):
    events_l = []
    for i in range(len(board)):
        new_board = TTT4()
        new_board.board = board[:]
        new_board.player = player
        ret = new_board.play(i)
        if 'invalid' in ret or 'win' in ret or 'draw' in ret:
            events_l.append([state, i, reward, None, True][:])
            continue
        next_state, next_ret = get_next_state(new_board.board[:], int(new_board.player), ai)
        reward = get_reward(ret, next_ret)
        events_l.append([state, i, reward, next_state, False][:])
    return events_l


def run(games=16):
    student = Agent(17, 16, model_name=name)
    game_n = 0
    while True:
        for _ in range(games):
            board = TTT4()
            end = False
            game_n += 1
            turn = 0
            while not end:
                player_turn = int(board.player)
                current_board = board.board[:]
                state = get_state(current_board, player_turn)
                if turn == 0:
                    action = random.randrange(16)
                else:
                    action = student.act(np.array(state))
                ret = board.play(action)
                events = get_events(state, current_board, player_turn, student.model)
                student.memory.append(events)
                print(f'{game_n}: {board.played}/16 {action} {ret}: player {player_turn-1}')
                board.print_board()
                if 'invalid' in ret:
                    break

                if 'win' in ret or 'draw' in ret:
                    end = True
                turn += 1
        if len(student.memory) > 5000:
            student.exp_replay()
        if game_n % 80 == 0:
            student.model.save(f'keras_model/{name}_{str(int(game_n))}')


name = 'base'
run()

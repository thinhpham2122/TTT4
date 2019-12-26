from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from time import time


class Agent:
    def __init__(self, state_size, action_size, model_name=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.inventory = []
        self.model_name = model_name
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99996
        self.epsilon_min = .25
        if model_name:
            try:
                print('loading model')
                self.model = load_model(f'keras_model/{model_name}')
            # self.epsilon = 0
            except:
                print('fail to load model, creating new model')
                self.model = self.model()
        else:
            print('creating new model')
            self.model = self.model()

    def model(self):
        model = Sequential()
        model.add(Dense(units=256, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if random.random() <= self.epsilon:
                empty_index = []
                for i in range(16):
                    if not state[0][i]:
                        empty_index.append(i)
                return empty_index[random.randrange(len(empty_index))]
        output = self.model.predict(state)
        print(np.round(output[0][0:4], 2))
        print(np.round(output[0][4:8], 2))
        print(np.round(output[0][8:12], 2))
        print(np.round(output[0][12:16], 2))
        return np.argmax(output)

    def exp_replay(self):
        states = []
        target_fs = []
        next_states = []
        current_states = []
        for event in self.memory:
            current_states.append(event[0][0][0])
            for [_, _, _, next_state, done] in event:
                if not done:
                    next_states.append(next_state[0])
        next_outputs = self.model.predict(np.array(next_states)).tolist()
        state_outputs = self.model.predict(np.array(current_states)).tolist()
        for e, event in enumerate(self.memory):
            state = event[0][0]
            target_f = state_outputs.pop(0)
            for i, [_, action, reward, _, done] in enumerate(event):
                if done:
                    target = reward
                else:
                    target = min(reward + (self.gamma * max(next_outputs.pop(0))), 1)
                target_f[action] = target
                if i == 15:
                    states.append(np.array(state[0][:]))
                    target_fs.append(np.array(target_f[:]))
        self.model.fit([states], [target_fs], epochs=1, verbose=1, batch_size=256)

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
	def __init__(self, state_size, action_size, model_name=None):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.inventory = []
		self.model_name = model_name

		self.gamma = 0.9
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.epsilon_min = .25
		if model_name:
			print('loading model')
			self.model = load_model(f'keras_model/{model_name}')
		else:
			print('fail to load model, creating new model')
			self.model = self.model()

	def model(self):
		model = Sequential()
		model.add(Dense(units=16, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=256, activation="relu"))
		model.add(Dense(units=256, activation="relu"))

		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))
		return model

	def act(self, state):
		# print(self.epsilon , self.epsilon_min)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
			if random.random() <= self.epsilon:
				return random.randrange(self.action_size)
		option = self.model.predict(state)
		print(np.round(option[0][0:4]))
		print(np.round(option[0][4:8]))
		print(np.round(option[0][8:12]))
		print(np.round(option[0][12:16]))

		return np.argmax(option)

	def exp_replay(self):
		states = []
		target_fs = []
		for event in self.memory:
			state = event[0][0]
			target_f = self.model.predict(state)
			for i, [_, action, reward, next_state, done] in enumerate(event):
				if done:
					target = reward
				else:
					target = reward  # + (self.gamma * max(self.model.predict(next_state)[0]))
				target_f[0][action] = target
				if i == 15:
					states.append(np.array(state[0][:]))
					target_fs.append(np.array(target_f[0][:]))
			# print(state, target_f)
		self.model.fit([states], [target_fs], epochs=1, verbose=2, batch_size=64)

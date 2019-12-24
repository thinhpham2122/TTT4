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
		self.memory = deque(maxlen=15000)
		self.high_memory = deque(maxlen=15000)
		self.inventory = []
		self.model_name = model_name

		self.gamma = 0.9
		self.epsilon = 1.0
		self.epsilon_decay = 0.9999
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
		model.add(Dense(units=16, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=256, activation="relu"))
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
		option = self.model.predict(state)
		print(np.round(option[0][0:4], 2))
		print(np.round(option[0][4:8], 2))
		print(np.round(option[0][8:12], 2))
		print(np.round(option[0][12:16], 2))

		return np.argmax(option)

	def exp_replay(self):
		states = []
		target_fs = []
		for event in self.high_memory:
			state = event[0][0]
			target_f = self.model.predict(state)
			max_reward = 0
			for i, [_, action, reward, next_state, done] in enumerate(event):
				max_reward = reward if reward > max_reward else max_reward
				if done:
					target = reward
				else:
					target = min(reward + (self.gamma * max(self.model.predict(next_state)[0])), 1)
				target_f[0][action] = target
				if i == 15:
					states.append(np.array(state[0][:]))
					target_fs.append(np.array(target_f[0][:]))

		for event in self.memory:
			state = event[0][0]
			target_f = self.model.predict(state)
			max_reward = 0
			for i, [_, action, reward, next_state, done] in enumerate(event):
				# print(state, action, next_state)
				max_reward = reward if reward > max_reward else max_reward
				if done:
					target = reward
				else:
					target = min(reward + (self.gamma * max(self.model.predict(next_state)[0])), 1)
				target_f[0][action] = target
				if i == 15:
					states.append(np.array(state[0][:]))
					target_fs.append(np.array(target_f[0][:]))
			if max_reward > 0.5:
				self.high_memory.append(event)
		# for i in zip(states, target_fs):
		# 	print(i)
		print(f'memory:{len(self.memory)} high:{len(self.high_memory)}')
		self.model.fit([states], [target_fs], epochs=1, verbose=2, batch_size=256)

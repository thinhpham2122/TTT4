from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
	def __init__(self, state_size, action_size, model_name=""):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name

		self.gamma = 0.5
		self.epsilon = 1.0
		self.epsilon_min = 0.25
		self.epsilon_decay = 0.9995
		self.model = load_model("keras_models/" + model_name) if model_name else self.model()

	def model(self):
		model = Sequential()
		model.add(Dense(units=20, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=20, activation="relu"))
		model.add(Dense(units=20, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
			if random.random() <= self.epsilon:
				return random.randrange(self.action_size)
		return np.argmax(self.model.predict(state))

	def exp_replay(self):
		states = []
		target_fs = []
		mini_batch = []

		l = len(self.memory)
		pick = np.random.choice(l, 100 if 100 < l else l)
		for i in pick:
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state in mini_batch:
			target = reward + self.gamma * max(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			states.append(np.array(state[0][:]))
			target_fs.append(np.array(target_f[0][:]))
		self.model.fit([states], [target_fs], epochs=1, verbose=0, batch_size=256)

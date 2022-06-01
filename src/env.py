import gym
import numpy as np

from battery import Battery
from memory import Memory

class Grid(gym.Env):
    def __init__(self, config, data):
        super().__init__()

        self.memory_size = config['memory_size']
        self.timesteps = config['timesteps']
        self.pretrain_length = config['pretrain_length']
        self.action_size = config['action_size']

        self.data = data['x']
        self.sc_energy = data['scaler_energy']
        self.sc_price = data['scaler_price']
        self.dlength = len(self.data)

        self.battery = Battery(self.action_size, self.sc_energy, self.sc_price)
        self.memory = Memory(self.memory_size)

    def _get_state(self, day, hour, soc):
        nonzeros = self.historical_price[self.historical_price != 0]
        average_price = 0
        if nonzeros.size > 0:
            average_price = np.mean(nonzeros)
        state = np.concatenate((self.data[day*24+hour, :], soc, np.array([average_price])), axis=-1)
        return state

    def reset(self):
        self.SOC = np.array([self.battery.initial_SOC])
        self.historical_price = np.zeros(self.timesteps)
        self.day = 0
        self.hour = 0
        self.timestep = 0
        state = self._get_state(self.day, self.hour, self.SOC)
        return state

    def step(self, action):
        self.historical_price[self.timestep] = self.data[self.day*24 + self.hour, 2]
        state = self._get_state(self.day, self.hour, self.SOC)
        next_SOC, reward, infos = self.battery.compute(state, action)
        done = False

        self.timestep = (self.timestep + 1) % self.timesteps
        self.hour = (self.hour + 1) % 24
        self.day += 1 if self.hour == 0 else 0

        self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
        next_state = self._get_state(self.day, self.hour, next_SOC)
        
        if self.day * 24 + self.hour == self.dlength - 1:
            done = True
        experience = state, action, reward, next_state, done
        return experience, infos
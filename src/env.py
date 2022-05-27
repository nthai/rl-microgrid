import gym
import numpy as np

from battery import Battery
from memory import Memory

class DayCountExceeded(Exception): pass

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
        if self.day >= self.dlength / 24:
            raise DayCountExceeded(f'Day count ({self.day}) has exceeded data length ({self.dlength / 24})')
        self.historical_price[self.timestep] = self.data[self.day*24 + self.hour, 2]
        state = self._get_state(self.day, self.hour, self.SOC)
        next_SOC, reward = self.battery.compute(state, action)
        done = False

        self.timestep = (self.timestep + 1) % self.timesteps
        if self.hour < 23:
            self.hour += 1
            self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
            next_state = self._get_state(self.day, self.hour, next_SOC)
        else:
            done = True
            self.day += 1
            self.hour = 0
            if self.day >= self.dlength / 24:
                raise DayCountExceeded(f'Day count ({self.day}) has exceeded data length ({self.dlength / 24})')
            else:
                self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
                next_state = self._get_state(self.day, self.hour, next_SOC)
        experience = state, action, reward, next_state, done
        self.memory.store(experience)
        return experience

    def init_memory(self):
        '''params:
            - data: np.ndarray containing [pv, load, price]
        '''
        self.reset()
        SOC = np.array([self.battery.initial_SOC])
        for _ in range(self.pretrain_length):
            self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
            # average_price = np.mean(np.array([price for price in self.historical_price if price != 0]))
            average_price = self.historical_price[self.historical_price != 0].mean()

            state = np.concatenate((self.data[self.day * 24 + self.hour, :], SOC,
                                    np.array([average_price])), axis = -1)
            action = np.random.randint(0, self.action_size)
            next_SOC, reward = self.battery.compute(state, action)
            self.timestep = (self.timestep + 1) % self.timesteps

            done = False
            # TODO: simplify this part
            if self.hour < 23:
                self.hour += 1
                self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
                # average_price = np.mean(np.array([price for price in self.historical_price if price != 0]))
                average_price = self.historical_price[self.historical_price != 0].mean()
                next_state = np.concatenate((self.data[self.day * 24 + self.hour, :], next_SOC,
                                             np.array([average_price])), axis=-1)
            else:
                done = True
                self.day += 1
                self.hour = 0
                if self.day >= self.dlength / 24:
                    break
                else:
                    self.historical_price[self.timestep] = self.data[self.day * 24 + self.hour, 2]
                    # average_price = np.mean(np.array([price for price in self.historical_price if price != 0]))
                    average_price = self.historical_price[self.historical_price != 0].mean()
                    next_state = np.concatenate((self.data[self.day * 24 + self.hour, :], next_SOC,
                                                np.array([average_price])), axis=-1)
            experience = state, action, reward, next_state, done
            # print(experience)
            self.memory.store(experience)
            
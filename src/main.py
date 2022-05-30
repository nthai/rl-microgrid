import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from env import DayCountExceeded, Grid
from dqn import DQNAgent

DATAFILE = '../Final Modified Data_Rev2.csv'
# DATAFILE = '../short.csv'

def data_read(filename: str) -> pd.DataFrame:
    df_raw = pd.read_csv(filename)
    df = df_raw.copy()
    df = df.iloc[:,1:]
    return df

def standardization(df: pd.DataFrame, show=True):
    # MinMaxScaler

    # The mean is not shifted to zero-centered
    sc_price = StandardScaler(with_mean = False)
    price = sc_price.fit_transform(df.iloc[:, 2:].values)

    sc_energy = StandardScaler(with_mean = False)
    pv = sc_energy.fit_transform(df.iloc[:, 0:1].values)
    load = sc_energy.transform(df.iloc[:, 1:2].values)

    x = np.concatenate([pv, load, price], axis=-1)

    if show:
        _, ax = plt.subplots(3, 2, figsize = (12, 12))

        ax[0, 0].plot(x[:, 0])
        ax[0, 0].set_xlabel("Hour")
        ax[0, 0].set_ylabel("PV (kW)")

        ax[0, 1].plot(x[0:24, 0])
        ax[0, 1].set_xlabel("Hour")
        ax[0, 1].set_ylabel("PV (pu)")

        ax[1, 0].plot(x[:, 1])
        ax[1, 0].set_xlabel("Hour")
        ax[1, 0].set_ylabel("Load (kW)")

        ax[1, 1].plot(x[0:24, 1])
        ax[1, 1].set_xlabel("Hour")
        ax[1, 1].set_ylabel("Load (pu)")

        ax[2, 0].plot(x[:, 2])
        ax[2, 0].set_xlabel("Hour")
        ax[2, 0].set_ylabel("Price ($/kWh)")

        ax[2, 1].plot(x[0:24, 2])
        ax[2, 1].set_xlabel("Hour")
        ax[2, 1].set_ylabel("Price (pu)")
        plt.savefig('figures/standardized.png', format='png', dpi=600)

    data = {
        'scaler_energy': sc_energy,
        'scaler_price': sc_price,
        'x': x
    }
    return data

def get_config() -> dict:
    config = {
        'state_size': 5, # PV, load, RTP, past 24 hour average RTP, SOC
        'action_size': 11,
        'learning_rate': .001,
        'episodes': 5,
        'batch_size': 64,
        'timesteps': 24,
        'explore_start': 1.0, # exploration probability at start
        'explore_stop': .01, # minimum exploration probability
        'decay_rate': .0001, # exponential decay rate for exploration prob
        'gamma': .95, # Discounting rate of future reward
        'pretrain_length': 10000, # # of experiences stored in Memory during initialization
        'memory_size': 10000 # # of experiences Memory can keep
    }

    return config

def train(data: dict, config: dict):

    env = Grid(config, data)
    agent = DQNAgent(config)
    print(env.dlength)
    env.init_memory()

    for ep in range(config['episodes']):
        print(f'Episode {ep} starts.')
        state = env.reset()
        tstart = time.time()
        total_reward = 0
        while True:
            try:
                action = agent.get_action(state)
                _, _, reward, state, _ = env.step(action)
                total_reward += reward[0]
                agent.train(env.memory)
            except DayCountExceeded:
                break
            except IndexError as err:
                print(action)
                raise err

        tend = time.time()
        print(f'Episode {ep} - training time: {(tend - tstart)/60:.2f}mins')
        print(f'Episode {ep} - total reward: {total_reward:.2f}')

def main():
    df = data_read(DATAFILE)
    config = get_config()

    if 'short' in DATAFILE:
        print(f'Working on smaller dataset: {DATAFILE}')
        config['memory_size'] = 100

    data = standardization(df, show=False)
    train(data, config)

if __name__ == '__main__':
    if __debug__:
        np.seterr(all='warn')
        import warnings
        warnings.filterwarnings('error')

    np.random.seed(42)
    main()

import numpy as np
from memory import Memory

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

class DQNNet():
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_model()
    
    def create_model(self) -> Model:
        input = Input(shape = (self.state_size, ))
        x = Dense(40, activation="elu", 
                kernel_initializer=glorot_uniform(seed = 42))(input)
        # x = Dense(160, activation="elu",
        #         kernel_initializer=glorot_uniform(seed = 42))(x)
        output = Dense(self.action_size, activation = "linear", 
                kernel_initializer=glorot_uniform(seed = 42))(x)
        model = Model(inputs=[input], outputs=[output])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def save(self, path='models/model_weights.ckpt'):
        self.model.save_weigths(path)
    
    def load(self, path='models/model_weights.ckpt'):
        self.model.load_weights(path)

class DQNAgent():
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']

        self.explore_start = config['explore_start']
        self.explore_stop = config['explore_stop']
        self.decay_rate = config['decay_rate']

        self.net = DQNNet(self.state_size, self.action_size, self.learning_rate)
        # prioritized experience replay buffer
        self.memory = Memory(config['memory_size'])
        self.reset()

    def store(self, prev_state, prev_mask, action, prob, reward,
              next_state, done):
        experience = (prev_state, action, reward, next_state, done)
        self.memory.store(experience)

    def reset(self):
        self.decay_step = 0

    def select_action(self, state):
        epsilon = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)
        qfactors = self.net.model.predict(np.expand_dims(state, axis = 0), verbose=0)
        action = np.argmax(qfactors)
        if np.random.random() < epsilon:
            # explore
            action = np.random.randint(0, self.action_size)
        self.decay_step += 1
        return action, qfactors[0][action]
    
    def train(self):
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_size)

        try:
            states_mb = np.array([each[0][0] for each in batch])
            actions_mb = np.array([each[0][1] for each in batch])
            rewards_mb = np.array([each[0][2][0] for each in batch]) 
            next_states_mb = np.array([each[0][3] for each in batch])
            dones_mb = np.array([each[0][4] for each in batch])
        except:
            print([each[0][0] for each in batch])
            raise

        nextq = self.net.model.predict(next_states_mb, verbose=0)
        maxq = np.amax(nextq, axis=1)
        td_target = self.net.model.predict(states_mb, verbose=0)

        td_target[range(self.batch_size), actions_mb] = \
            rewards_mb + (1-dones_mb)*maxq*self.gamma
        
        self.net.model.fit(states_mb, td_target, sample_weight=ISWeights_mb.ravel(), epochs=1, verbose=0)

        # Update priority
        errs = []
        preds = self.net.model.predict(states_mb, verbose=0)
        for idx in range(self.batch_size):
            err = np.abs(preds[idx][actions_mb[idx]] - td_target[idx][actions_mb[idx]])
            errs.append(err)
        errs = np.array(errs)
        self.memory.batch_update(tree_idx.astype(int), errs)

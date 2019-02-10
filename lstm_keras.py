from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class Input(object):
    def __init__(self, data, n_steps, batch_size, skip_step=5):
        self.data = data
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.current_idx = 0
        self.skip_step = skip_step #We generated fifth coord, we can skip five to generate next



class Model(object):
    def __init__(self, hidden_size, timesteps, input_dim, is_training):
        self.model = Sequential()
        self.model.add(LSTM(hidden_size, input_shape=(timesteps, input_dim), 
                    return_sequence=True))
        self.model.add(LSTM(hidden_size, return_sequence=False))
        self.model.add(Dense(1, activation="relu"))
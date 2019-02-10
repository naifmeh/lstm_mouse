from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

class Input(object):
    def __init__(self, data, n_steps, batch_size, skip_step=5):
        self.data = data
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.current_idx = 0
        self.skip_step = skip_step #We generated fifth coord, we can skip five to generate next

def 

class Model(object):
    def __init__(self, hidden_size, timesteps, input_dim, is_training, output_dim):
        self.model = Sequential()
        self.model.add(LSTM(hidden_size, input_shape=(timesteps, input_dim), 
                    return_sequence=True))
        self.model.add(LSTM(hidden_size, return_sequence=True))
        self.model.add(TimeDistributed(Dense(output_dim, activation="relu")))
        if is_training:
            self.model.add(Dropout(0.3))

        self.model.summary()

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

        checkpointer = ModelCheckpoint(filepath=__dirname__+'/model-{epoch:02d}.hdf5', verbose=1)

    def train(train_x, train_y, num_steps, batch_size, epochs, valid_data, valid_steps):
        self.model.fit(train_x, train_y, epochs=epochs, steps_per_epoch=num_steps,
                    validation_data=valid_data, valid_steps=valid_steps)

    def test(model_path):
        return
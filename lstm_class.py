def batch_producer(raw_data, batch_size, num_steps):
    """
    """



class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epochs_size = ((len(data) // batch_size) -1) // num_steps
        self.input_data, self targets = batch_producer(data, batch_size, num_steps)

class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size=2, num_layers,
                dropout=0.5, init_scale=0.05)

        coord_size = 2

        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps

        inputs = self.input_obj.input_data
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)
        
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, hidden_size]) #h and s vectors (2) but should I put 4 as i output a tuple
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)]
        )

        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)

        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True) #Same conf for each layer

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        output = tf.reshape(output, [-1, hidden_size]) #Second axis reshaping, the rest is up to tf

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, coord_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([coord_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, coord_size])

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_accross_timesteps=True, #HIghly time related data but not sure of this choice tho
            average_accross_batch=False
        )

        self.cost = tf.reduce_sum(loss)
        



        

        
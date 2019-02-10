import tensorflow as tf
import numpy as np

def batch_producer(raw_data, batch_size, num_steps):
    """
    """
    # TODO: Write this function

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

        self.dense = tf.contrib.layers.fully_connected(
            output,
            1,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.glorot_uniform_initializer,
        )

       

        # softmax_w = tf.Variable(tf.random_uniform([hidden_size, coord_size], -init_scale, init_scale))
        # softmax_b = tf.Variable(tf.random_uniform([coord_size], -init_scale, init_scale))
        # logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # logits = tf.reshape(logits, [self.batch_size, self.num_steps, coord_size])

        # loss = tf.contrib.seq2seq.sequence_loss(
        #     self.relu_out,
        #     self.input_obj.targets,
        #     tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
        #     average_accross_timesteps=True, #HIghly time related data but not sure of this choice tho
        #     average_accross_batch=False
        # )

        loss = tf.losses.mean_squared_error(
            dense,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
        )
        self.cost = tf.reduce_sum(loss)
        
        # self.relu_out = tf.nn.relu(tf.reshape(logits, [-1, coord_size]))
        # #self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, coord_size]))
        # self.predict = tf.cast(self.relu_out, tf.int32)
        correct_prediction = tf.equal(self.dense, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return
        
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradient(self.cost, tvars), 5)
        optimizer = tf.train.AdamOtimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        )

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update)

def train(train_data, num_epochs, num_layer, batch_size, model_save_name,
         learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93):

    training_input = Input(btach_size=batch_size, num_steps=15, data=train_data)
    model = Model(training_input, is_training=True, hidden_size=128, num_layers=num_layer)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver=tf.train.Saver()
        for epoch in range(num_epoch):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            model.assign_lr(sess, learning_rate * new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, model.hidden_size))
            for step in range(training_input.epoch_size):
                if step % 50 != 0:
                    cost, _, current_state = sess.run([model.cost, model.train_op, model.state])
                else:
                    cost, _, current_state, acc = sess.run([model.cost, model.train_op, model.state, model.accuracy])
                    print("Epoch {}, Step {}, cost {:.3f}, accuracy: {:.3f}".format(epoch, step, cost, acc))
            
            saver.save(sess, ".\\"+model_save_name, global_step=epoch)
        saver.save(sess, '.\\'+model_save_name+'-final')
        coord.request_stop()
        coord.join(threads)       
        
def test(model_path, test_data):
    test_input = Input(batch_size=10, num_steps=35, data=test_data)
    model = Model(test_input, is_training=False, hidden_size=128, num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((num_layers, 2, model.batch_size, model.hidden_size))

        saver.restore(sess, model_path)

        num_acc_batches = 20
        check_batch_idx = 15
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_val, pred, current_state, acc = sess.run([model.input_obj.targets, model.dense, model.state, model.accuracy])
                print('Predicted {}'.format(pred))
                print('True value {}'.format(true_val))
            else:
                acc, current_state = sess.run([model.accuracy, model.state])
            
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy is : {:.3f}".format(accuracy/(num_acc_batches-acc_check_thresh)))
        coord.request_stop()
        coord.join(threads)


        

        
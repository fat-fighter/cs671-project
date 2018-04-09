import tensorflow as tf


class MatchLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, encodings, question_masks):
        self.input_size = input_size
        self._state_size = state_size
        self.encodings = encodings
        self.question_masks = question_masks

        self.init_properties()

    def init_properties(self):
        self.zero_initializer = tf.initializers.zeros()
        self.iden_initializer = tf.initializers.identity()
        self.unit_initializer = tf.initializers.uniform_unit_scaling()

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, input, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            batch_size = tf.shape(self.encodings)[0]

            W_q = tf.get_variable(
                'W_q',
                [self.input_size, self.input_size],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )
            W_c = tf.get_variable(
                'W_c',
                [self.input_size, self.input_size],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )
            W_r = tf.get_variable(
                'W_r',
                [self._state_size, self.input_size],
                dtype=tf.float32,
                initializer=self.iden_initializer
            )
            W_a = tf.get_variable(
                'W_a',
                [self.input_size, 1],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )

            b_g = tf.get_variable(
                'b_g',
                [self.input_size],
                tf.float32,
                initializer=self.zero_initializer
            )
            b_a = tf.get_variable(
                'b_a',
                [1],
                tf.float32,
                initializer=self.zero_initializer
            )

            wq_e = tf.tile(
                tf.expand_dims(W_q, axis=0),
                [batch_size, 1, 1]
            )
            g = tf.tanh(
                tf.matmul(self.encodings, wq_e) + tf.expand_dims(
                    tf.matmul(input, W_c) + tf.matmul(state, W_r) + b_g, axis=1
                )
            )

            wa_e = tf.tile(
                tf.expand_dims(W_a, axis=0),
                [batch_size, 1, 1]
            )
            a = tf.nn.softmax(tf.squeeze(
                tf.matmul(g, wa_e) + b_a, axis=2)
            )
            a = tf.multiply(a, self.question_masks)

            question_attend = tf.reduce_sum(
                tf.multiply(self.encodings, tf.expand_dims(a, axis=2)),
                axis=1
            )
            z = tf.concat([input, question_attend], axis=1)

            W_f = tf.get_variable(
                'W_f',
                [self._state_size, self._state_size],
                dtype=tf.float32,
                initializer=self.iden_initializer
            )
            U_f = tf.get_variable(
                'U_f',
                [2 * self.input_size, self._state_size],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )
            b_f = tf.get_variable(
                'b_f',
                [self._state_size],
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.0)
            )

            W_z = tf.get_variable(
                'W_z',
                [self._state_size, self._state_size],
                dtype=tf.float32,
                initializer=self.iden_initializer
            )
            U_z = tf.get_variable(
                'U_z',
                [2 * self.input_size, self._state_size],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )
            b_z = tf.get_variable(
                'b_z',
                [self._state_size],
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.0)
            )

            W_o = tf.get_variable(
                'W_o',
                [self._state_size, self._state_size],
                dtype=tf.float32,
                initializer=self.iden_initializer
            )
            U_o = tf.get_variable(
                'U_o',
                [2 * self.input_size, self._state_size],
                dtype=tf.float32,
                initializer=self.unit_initializer
            )
            b_o = tf.get_variable(
                'b_o',
                [self._state_size],
                dtype=tf.float32,
                initializer=self.zero_initializer
            )

            z_t = tf.nn.sigmoid(
                tf.matmul(z, U_z) + tf.matmul(state, W_z) + b_z
            )
            f_t = tf.nn.sigmoid(
                tf.matmul(z, U_f) + tf.matmul(state, W_f) + b_f
            )
            o_t = tf.nn.tanh(
                tf.matmul(z, U_o) + tf.matmul(f_t * state, W_o) + b_o
            )

            output = z_t * state + (1 - z_t) * o_t
            new_state = output

        return output, new_state

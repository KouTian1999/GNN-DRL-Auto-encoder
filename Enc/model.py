import numpy as np
import tensorflow as tf

from utils import find_neighbors


class MLP(tf.keras.layers.Layer):
    def __init__(self, units, activations):
        super().__init__()
        self.num_units = units
        self.activations = activations

    def build(self, input_shape):
        self.layers = []
        for i, units in enumerate(self.num_units):
            self.layers.append(tf.keras.layers.Dense(units, self.activations[i]))

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class ActorGnn(tf.keras.Model):
    def __init__(self, m_b, n_b, emb_num, msg_num, gnn_layers, mlp_layers, mlp_units, act):
        super().__init__()
        self.m_b = m_b
        self.n_b = n_b
        self.emb_num = emb_num
        self.msg_num = msg_num

        self.gnn_layers = gnn_layers

        self.mlp_layers = mlp_layers
        self.units_num = mlp_units
        self.activation = act

        self.pos = np.arange(self.m_b * self.n_b, dtype=int)
        self.up_nei, self.down_nei, self.left_nei, self.right_nei = find_neighbors(self.pos, self.m_b, self.n_b)

        self.flat = tf.keras.layers.Flatten()
        self.in_layer = tf.keras.layers.Dense(self.emb_num)

        units = [self.units_num] * (self.mlp_layers - 1) + [self.msg_num]
        activations = [self.activation] * (self.mlp_layers - 1) + [None]

        self.msg_row_nn = MLP(units, activations)
        self.msg_col_nn = MLP(units, activations)

        units[-1] = self.emb_num
        self.emb_nn = MLP(units, activations)

        self.ro_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def mat_to_emb(self, state):
        return self.in_layer(tf.expand_dims(self.flat(state), axis=-1))

    def emb_to_act(self, emb):
        # act [batch_size, m_b * n_b]
        act = tf.squeeze(self.ro_layer(emb), axis=-1)
        #act = tf.math.round(act)
        batch_size = tf.shape(act)[0]
        act = tf.reshape(act, [batch_size, self.m_b, self.n_b])
        return act

    def call(self, state, batch=False):
        # state [(batch size), m_b, n_b]
        # if len(tf.shape(state)) == 2 and tf.shape(state)[0] == self.m_b and tf.shape(state)[1] == self.n_b:
        #     # state [batch size >= 1, m_b, n_b]
        if batch is False and len(tf.shape(state)) == 2:
            state = tf.expand_dims(state, axis=0)

        # h_emb [batch_size, m_b * n_b, num_emb]
        h_emb = self.mat_to_emb(state)

        for i in range(self.gnn_layers):
            h_emb_up = tf.gather(h_emb, self.up_nei, axis=1)
            h_emb_down = tf.gather(h_emb, self.down_nei, axis=1)
            h_emb_left = tf.gather(h_emb, self.left_nei, axis=1)
            h_emb_right = tf.gather(h_emb, self.right_nei, axis=1)

            # msg [batch, m_b * n_b, num_msg]
            m_up = self.msg_col_nn(tf.concat([h_emb, h_emb_up], axis=-1))
            m_down = self.msg_col_nn(tf.concat([h_emb, h_emb_down], axis=-1))
            m_left = self.msg_row_nn(tf.concat([h_emb, h_emb_left], axis=-1))
            m_right = self.msg_row_nn(tf.concat([h_emb, h_emb_right], axis=-1))

            # aggregation : sum
            # m_agg [batch, m_b * n_b, num_msg]
            m_agg = m_up + m_down + m_left + m_right

            # h_emb_updated [batch_size, m_b * n_b, num_emb]
            h_emb = self.emb_nn(tf.concat([h_emb, m_agg], axis=-1))

        # act [batch_size, m_b, n_b]
        act = self.emb_to_act(h_emb)

        # if tf.shape(act)[0] == 1 and tf.shape(act)[1] == self.m_b and tf.shape(act)[2] == self.n_b:
        if batch is False and len(tf.shape(state)) == 3:
            act = tf.squeeze(act, axis=0)

        return act


class CriticGNN(tf.keras.Model):
    def __init__(self, m_b, n_b, emb_num, msg_num, gnn_layers, mlp_layers, mlp_units, act):
        super().__init__()
        self.m_b = m_b
        self.n_b = n_b
        self.emb_num = emb_num
        self.msg_num = msg_num

        self.mlp_layers = mlp_layers
        self.units_num = mlp_units
        self.activation = act

        self.gnn_layers = gnn_layers

        self.pos = np.arange(self.m_b * self.n_b)
        self.up_nei, self.down_nei, self.left_nei, self.right_nei = find_neighbors(self.pos, self.m_b, self.n_b)

        self.flat = tf.keras.layers.Flatten()
        self.in_layer = tf.keras.layers.Dense(self.emb_num)

        units = [self.units_num] * (self.mlp_layers - 1) + [self.msg_num]
        activations = [self.activation] * (self.mlp_layers - 1) + [None]

        self.msg_row_nn = MLP(units, activations)
        self.msg_col_nn = MLP(units, activations)

        units[-1] = self.emb_num
        self.emb_nn = MLP(units, activations)

        self.ro_layer = tf.keras.layers.Dense(1)
        self.ro_layer_2 = tf.keras.layers.Dense(1)

    def mat_to_emb(self, state, actor):
        state = tf.expand_dims(self.flat(state), axis=-1)
        actor = tf.expand_dims(self.flat(actor), axis=-1)
        return self.in_layer(tf.concat([state, actor], axis=-1))

    def emb_to_q(self, emb):
        # emb [batch_size, m_b * n_b, num_emb]
        q_value = tf.squeeze(self.ro_layer(emb), axis=-1)  # [batch_size, m_b * n_b]
        q_value = self.ro_layer_2(q_value)  # [batch_size, 1]
        return q_value

    def call(self, state, actor, batch=False):
        # state [(batch size), m_b, n_b]
        # actor [(batch size), m_b, n_b]
        #if len(tf.shape(state)) == 2 and tf.shape(state)[0] == self.m_b and tf.shape(state)[1] == self.n_b:
            # state [batch size >= 1, m_b, n_b]
        if batch is False and len(tf.shape(state)) == 2:
            state = tf.expand_dims(state, axis=0)

        if batch is False and len(tf.shape(actor)) == 2:
            # actor [batch size >= 1, m_b, n_b]
            actor = tf.expand_dims(actor, axis=0)

        # h_emb [batch_size, m_b * n_b, num_emb]
        h_emb = self.mat_to_emb(state, actor)

        for i in range(self.gnn_layers):
            h_emb_up = tf.gather(h_emb, self.up_nei, axis=1)
            h_emb_down = tf.gather(h_emb, self.down_nei, axis=1)
            h_emb_left = tf.gather(h_emb, self.left_nei, axis=1)
            h_emb_right = tf.gather(h_emb, self.right_nei, axis=1)

            # msg [batch, m_b * n_b, num_msg]
            m_up = self.msg_col_nn(tf.concat([h_emb, h_emb_up], axis=-1))
            m_down = self.msg_col_nn(tf.concat([h_emb, h_emb_down], axis=-1))
            m_left = self.msg_row_nn(tf.concat([h_emb, h_emb_left], axis=-1))
            m_right = self.msg_row_nn(tf.concat([h_emb, h_emb_right], axis=-1))

            # aggregation : sum
            # m_agg [batch, m_b * n_b, num_msg]
            m_agg = m_up + m_down + m_left + m_right

            # h_emb_updated [batch_size, m_b * n_b, num_emb]
            h_emb = self.emb_nn(tf.concat([h_emb, m_agg], axis=-1))

        # q_value [batch_size, 1]
        q_value = self.emb_to_q(h_emb)

        return q_value

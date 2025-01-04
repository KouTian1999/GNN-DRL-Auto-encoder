import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from utils import LinearEncoder, ebnodb2no, compute_bler, compute_ber

from ber_bp import atanh2, tanh1_2


# @tf.function
# def normalize(input):
#     mean = tf.reduce_mean(input, axis=1, keepdims=True)
#     std = tf.math.reduce_std(input, axis=1, keepdims=True)
#     return tf.math.divide_no_nan((input - mean), std)


@tf.function
def normalize_by_mean(input):
    mean = tf.reduce_mean(input, axis=1, keepdims=True)
    return tf.math.divide_no_nan(input, mean)


class NodeInfoUpdate(Layer):
  def __init__(self, output, gather_index):
    super().__init__()
    self._output = output
    self._gather_index = gather_index

  def build(self, input_shape):
    num_dims = input_shape[-1]

  def call(self, h, h_r, h_old):
    if self._output == 'E':
      h = tanh1_2(h)
    # h: h_m or h_e - tf.float32 [batch_size, num_edge, 1]
    # h_r - tf.float32 [batch_size, num_edge, 1]
    h = tf.transpose(h, (1,2,0))
    h_ragged = tf.gather(h, self._gather_index, axis=0)

    # Aggregation
    # h_agg - tf.float32 [batch_size, num_edge, 1]
    if self._output == 'E':
      h_agg = tf.reduce_prod(h_ragged, axis=1)
    else:
      # adding weights
      h_agg = tf.reduce_sum(h_ragged, axis=1)
      h_agg = h_agg + tf.transpose(h_r, (1,2,0))

    h_new = tf.transpose(h_agg, (2,0,1))

    # h_new = activation(h_agg)
    if self._output == 'E':
      h_new = atanh2(h_new)

    h_res = tf.abs(h_new - h_old)
    return h_new, h_res


class MLP(Layer):
    def __init__(self, units, activations, use_bias):
        super().__init__()
        self._num_units = units
        self._activations = activations
        self._use_bias = use_bias

    def build(self, input_shape):
        self._layers = []
        for i, units in enumerate(self._num_units):
            self._layers.append(Dense(units,
                                      self._activations[i],
                                      use_bias=self._use_bias[i]))

    def call(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs


class WBP_Decoder(Layer):
    def __init__(self, pcm, num_iter, multiloss=False):
        super().__init__()
        self._num_msg_dims = 1
        self._pcm = pcm  # Parity check matrix
        self._num_edges = int(np.sum(pcm))  # Number of edges
        # Array of shape [num_edges, 2]
        # 1st col = CN id, 2nd col = VN id
        # The ith row of this array defines the ith edge.
        self._rolled_edges = np.stack(np.where(pcm), axis=1)

        # hm_update_edges[i] contains the neighbour edge ids for updating h_m[i]
        hm_update_edges = []
        he_update_edges = []
        readout_edges = []
        for i in range(self._num_edges):
            cn = self._rolled_edges[i, 0]
            vn = self._rolled_edges[i, 1]
            hm_update_edges.append(np.where((self._rolled_edges[:, 1] == vn) & (self._rolled_edges[:, 0] != cn))[0])
            he_update_edges.append(np.where((self._rolled_edges[:, 1] != vn) & (self._rolled_edges[:, 0] == cn))[0])
            readout_edges.append(np.where((self._rolled_edges[:, 1] == vn))[0])
        self._hm_update_edges = tf.ragged.constant(hm_update_edges)
        self._he_update_edges = tf.ragged.constant(he_update_edges)
        self._readout_edges = tf.ragged.constant(readout_edges)

        vn_edges = []
        for i in range(pcm.shape[1]):
            vn_edges.append(np.where(self._rolled_edges[:, 1] == i)[0][0])
        self._vn_edges = tf.ragged.constant(vn_edges)

        self._num_iter = num_iter
        self._multiloss = multiloss

        # MLP
        self._num_mlp_layers = 3
        self._activation = 'elu'
        self._num_hidden_units = 32

    @property
    def num_iter(self):
        return self._num_iter

    @num_iter.setter
    def num_iter(self, value):
        self._num_iter = value

    def build(self, input_shape):
        num_dims = input_shape[-1]

        self.update_h_m = NodeInfoUpdate(output='M',
                                         gather_index=self._hm_update_edges)
        self.update_h_e = NodeInfoUpdate(output='E',
                                         gather_index=self._he_update_edges)
        self.readout = NodeInfoUpdate(output='LLR',
                                      gather_index=self._readout_edges)
        # w_MLP
        units = [self._num_hidden_units] * (self._num_mlp_layers - 1) + [self._num_msg_dims]
        activations = [self._activation] * (self._num_mlp_layers)
        use_bias = [True] * self._num_mlp_layers
        self.w_mlp = MLP(units, activations, use_bias)

    def call(self, llr):
        # llr - tf.float32 [batch_size, n]
        # initialize h_m, h_e, h_r [batch_size, num_edges, 1]
        batch_size = tf.shape(llr)[0]
        llr_received = tf.expand_dims(llr, -1)

        h_r = tf.gather(llr_received, self._rolled_edges[:, 1], axis=1)
        llr_gathered = tf.gather(llr_received, self._rolled_edges[:, 1], axis=1)
        h_m = tf.gather(llr_received, self._rolled_edges[:, 1], axis=1)
        h_e = tf.zeros([batch_size, self._num_edges, self._num_msg_dims])

        h_m_res = tf.zeros([batch_size, self._num_edges, self._num_msg_dims])
        llr_res = tf.zeros([batch_size, self._num_edges, self._num_msg_dims])

        llr_hat = []
        weights = []
        for i in range(self._num_iter):
            # Update E
            h_e, h_e_res = self.update_h_e(h_m, h_r, h_e)
            # adding weights to h_e

            feature = tf.concat([normalize_by_mean(tf.abs(h_e)), normalize_by_mean(h_e_res),
                                 normalize_by_mean(h_m_res), normalize_by_mean(llr_res)], axis=-1)
            w = self.w_mlp(feature)
            h_e = w * h_e

            weights.append(w[0, :, 0])

            # Update M
            h_m, h_m_res = self.update_h_m(h_e, h_r, h_m)

            # Readout
            llr_gathered, llr_res = self.readout(h_e, h_r, llr_gathered)
            llr_out = tf.gather(llr_gathered, self._vn_edges, axis=1)

            if self._multiloss:
                # final output: list of llr_hat - [tf.float32 [batch_size, n]]
                llr_hat.append(tf.squeeze(llr_out, -1))

        if not self._multiloss:
            # final output: llr_hat - tf.float32 [batch_size, n]
            # llr_hat = tf.squeeze(llr_out, -1)
            llr_hat.append(tf.squeeze(llr_out, -1))

        return llr_hat


class E2EModel_GNN(tf.keras.Model):
    def __init__(self, pcm, bp_inter):
        super().__init__()
        self._m = pcm.shape[0]
        self._n = pcm.shape[1]
        self._k = self._n - self._m
        self._decoder = WBP_Decoder(pcm, bp_inter)

        # self._binary_source = BinarySource()
        self._encoder = LinearEncoder(pcm, is_pcm=True)

    def __call__(self, batch_size, ebno_db):
        # batch_size: int []
        # ebno_db: float [batch_size, 1]

        bits = tf.cast(tf.random.uniform([batch_size, self._k], 0, 2, tf.int32), dtype=tf.float32)

        # Linear encoder: c = bits * gnm
        # codewords - tf.float32 [batch_size, n]
        codes = self._encoder(bits)

        # BPSK: 0 to 1, 1 to -1
        # modulated symbols - tf.float32 [batch_size, n]
        symbols = (-1) ** codes

        # Map SNR(dB) to noise variance no: no = (ebno * coderate * M)^(-1)
        # no - tf.float32 [batch_size, 1]
        if self._decoder is not None:
            no = ebnodb2no(ebno_db, 2, self._k / self._n)  # E.N WBP with received_llr = 2*received_symbols/no

        # AWGN Channel: received_symbols = symbols + noise, noise ~ (0, no)
        # received_symbols & received_llr  - tf.float32 [batch_size, n]
        noise = tf.sqrt(no) * tf.random.normal(tf.shape(symbols))
        received_symbols = symbols + noise
        # received_llr = 4*received_symbols/no
        received_llr = 2 * received_symbols / no

        # Decoder
        # llr - tf.float32 [batch_size, n]
        if self._decoder is not None:
            llr = self._decoder(received_llr)

        return codes, llr


def test_ber_gnn(pcm, bp_inter, snr_db, model_fp, max_batch=10000, block_per_batch=10000, num_target_block_errors=500):
    model = E2EModel_GNN(pcm, bp_inter)
    status = model.load_weights(model_fp)
    status.expect_partial()
    ber = 0
    bler = 0
    block_error = 0
    num_batch = 0
    for i in range(max_batch):
        c, llr_hat = model(block_per_batch, snr_db)
        c_hat = tf.cast(tf.less(llr_hat[-1], 0), tf.float32)
        # tf.print('c_hat', c_hat, summarize = -1)
        num_batch += 1
        ber += compute_ber(c, c_hat).numpy()
        bler += compute_bler(c, c_hat).numpy()
        block_error += compute_bler(c, c_hat).numpy() * block_per_batch
        if np.greater(block_error, num_target_block_errors):
            # runtime = time.perf_counter() - runtime_start
            break
    ber = ber / num_batch
    bler = bler / num_batch

    return ber, bler

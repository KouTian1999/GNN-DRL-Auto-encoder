import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.layers import Layer
from utils import LinearEncoder, BinarySource, compute_ber, compute_bler, ebnodb2no


@tf.function
def atanh2(input):
  E1 = 1 + input
  E1 = tf.clip_by_value(E1, clip_value_min = 1e-7, clip_value_max = 2-1e-7)
  E2 = 1 - input
  E2 = tf.clip_by_value(E2, clip_value_min = 1e-7, clip_value_max = 2-1e-7)
  return tf.math.log(E1/E2)


@tf.function
def tanh1_2(input):
  return tf.tanh(input/2)


class E2EModel(tf.keras.Model):
    """End-to-end channel coding model.

    Parameters
    ----------
    gnm: [k, n] numpy.array
          The generator matrix
    decoder: tf.keras.Layer
    """

    def __init__(self, gnm, decoder):
        super().__init__()
        self._gnm = gnm
        self._k = gnm.shape[0]
        self._n = gnm.shape[1]
        self._decoder = decoder

        self._binary_source = BinarySource()
        self._encoder = LinearEncoder(gnm)

    @tf.function()
    def call(self, batch_size, ebno_db):
        # batch_size: int []
        # ebno_db: float [batch_size, 1]

        # Generate random bits
        # source bits - tf.float32 [batch_size, k]
        bits = self._binary_source([batch_size, self._k])
        # bits = tf.zeros([batch_size, self._k])
        # bits = tf.cast(tf.random.uniform([batch_size, self._k], 0, 2, tf.int32), dtype=tf.float32)

        # Linear encoder: c = bits * gnm
        # codewords - tf.float32 [batch_size, n]
        codes = self._encoder(bits)

        # BPSK: 0 to 1, 1 to -1
        # modulated symbols - tf.float32 [batch_size, n]
        symbols = (-1) ** codes

        # Map SNR(dB) to noise variance no: no = (ebno * coderate * M)^(-1)
        # no - tf.float32 [batch_size, 1]
        if self._decoder is not None:
            # no = ebnodb2no(ebno_db, 1, self._k/self._n)
            # no = ebnodb2no(ebno_db, 1, 1) #chentao BP matlab with received_llr = 4*received_symbols/no
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

        # tf.print('bits', bits, summarize = -1)
        # tf.print('codes', codes, summarize = -1)
        # tf.print('symbols', symbols, summarize = -1)
        # tf.print('no', no, summarize = -1)
        # tf.print('noise', noise, summarize = -1)
        # tf.print('received_symbols', received_symbols, summarize = -1)
        # tf.print('received_llrs', received_llr, summarize = -1)
        # tf.print('llr', llr, summarize = -1)
        return codes, llr


class NodeInfoUpdate_BP(Layer):
    def __init__(self, operation, activation, gather_index):
        super().__init__()
        self._operation = operation
        self._activfunc = activation
        self._gather_index = gather_index

    def build(self, input_shape):
        num_dims = input_shape[-1]

    def call(self, h, h_r):
        # h: h_m or h_e - tf.float32 [batch_size, num_edge, 1]
        # h_r - tf.float32 [batch_size, num_edge, 1]
        h = tf.transpose(h, (1, 2, 0))
        h_ragged = tf.gather(h, self._gather_index, axis=0)

        # Aggregation
        # h_agg - tf.float32 [batch_size, num_edge, 1]
        if self._operation == 'prod':
            h_agg = tf.reduce_prod(h_ragged, axis=1)
        elif self._operation == 'sum':
            # adding weights
            h_agg = tf.reduce_sum(h_ragged, axis=1)
            h_agg = h_agg + tf.transpose(h_r, (1, 2, 0))

        h_new = tf.transpose(h_agg, (2, 0, 1))

        # h_new = activation(h_agg)
        if self._activfunc is not None:
            h_new = self._activfunc(h_new)

        return h_new


class BP_Decoder(Layer):
  def __init__(self, pcm, num_iter, multiloss=False):
    super().__init__()
    self._num_embed_dims = 1
    self._pcm = pcm # Parity check matrix
    self._num_edges = int(np.sum(pcm)) # Number of edges
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
      hm_update_edges.append(np.where((self._rolled_edges[:,1]== vn) & (self._rolled_edges[:,0] != cn))[0])
      he_update_edges.append(np.where((self._rolled_edges[:,1]!= vn) & (self._rolled_edges[:,0] == cn))[0])
      readout_edges.append(np.where((self._rolled_edges[:,1]== vn))[0])
    self._hm_update_edges = tf.ragged.constant(hm_update_edges)
    self._he_update_edges = tf.ragged.constant(he_update_edges)
    self._readout_edges = tf.ragged.constant(readout_edges)

    vn_edges = []
    for i in range(pcm.shape[1]):
      vn_edges.append(np.where(self._rolled_edges[:,1]==i)[0][0])
    self._vn_edges = tf.ragged.constant(vn_edges)

    self._num_iter = num_iter
    self._multiloss = multiloss

  @property
  def num_iter(self):
    return self._num_iter

  @num_iter.setter
  def num_iter(self, value):
    self._num_iter = value

  def build(self, input_shape):
    num_dims = input_shape[-1]

    self.update_h_m = NodeInfoUpdate_BP(operation='sum',
                                        activation= tanh1_2,
                                        gather_index= self._hm_update_edges)
    self.update_h_e = NodeInfoUpdate_BP(operation='prod',
                                        activation= atanh2,
                                        gather_index= self._he_update_edges)
    self.readout = NodeInfoUpdate_BP(operation='sum',
                                     activation = None,
                                     gather_index= self._readout_edges)

  def call(self, llr):
    # llr - tf.float32 [batch_size, n]
    # initialize h_m, h_e, h_r
    batch_size = tf.shape(llr)[0]
    llr = tf.expand_dims(llr, -1)
    h_r = tf.gather(llr, self._rolled_edges[:,1], axis=1)
    #h_m = tf.gather(llr, self._rolled_edges[:,1], axis=1)
    h_m = tanh1_2(h_r)
    h_e = tf.zeros([batch_size, self._num_edges, self._num_embed_dims])

    # BP iterations
    if self._multiloss:
      llr_hat = []
    for i in range(self._num_iter):
      # Update E
      h_e = self.update_h_e(h_m, h_r)
      # Update M
      h_m = self.update_h_m(h_e, h_r)
      # Readout
      llr_out = self.readout(h_e, h_r)
      llr_out = tf.gather(llr_out, self._vn_edges, axis =1)

      # tf.print('iteration', i+1, summarize = -1)
      # tf.print('E message', h_e, summarize = -1)
      # tf.print('M message', h_m, summarize = -1)
      # tf.print('R Message', h_r, summarize = -1)
      # tf.print('llr', llr_out, summarize = -1)

      if self._multiloss:
      # final output: list of llr_hat - [tf.float32 [batch_size, n]]
        llr_hat.append(tf.squeeze(llr_out, -1))

    if not self._multiloss:
    # final output: llr_hat - tf.float32 [batch_size, n]
      llr_hat = tf.squeeze(llr_out, -1)
    return llr_hat


def test_model_bp(model, params):

  snr_low, snr_high, max_batch, block_per_batch, num_target_block_errors = params

  snr_dbs = tf.expand_dims(tf.range(snr_low, snr_high + 1, dtype=tf.float32), -1)
  num_points = tf.shape(snr_dbs)[0]
  bers = np.zeros(num_points)
  blers = np.zeros(num_points)

  for i in range(num_points):
    runtime_start = time.perf_counter()
    ber = 0
    bler = 0
    block_error = 0
    num_batch = 0
    for ii in range(max_batch):
      c, llr_hat = model(block_per_batch, snr_dbs[i])
      c_hat = tf.cast(tf.less(llr_hat, 0), tf.float32)
      #tf.print('c_hat', c_hat, summarize = -1)
      num_batch += 1
      ber += compute_ber(c, c_hat).numpy()
      bler += compute_bler(c, c_hat).numpy()
      block_error += compute_bler(c, c_hat).numpy() * block_per_batch
      if np.greater(block_error, num_target_block_errors):
        runtime = time.perf_counter() - runtime_start
        break
    bers[i] = ber/num_batch
    blers[i] = bler/num_batch
    print(f"SNR {snr_dbs[i]}, ber = {bers[i]:.5e}, bler = {blers[i]:.5e}, " \
          f"block_num = {num_batch * block_per_batch:d}, run_time = {runtime:.2f}s")

  return bers, blers

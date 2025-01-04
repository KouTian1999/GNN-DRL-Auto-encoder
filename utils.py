import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import sys
import warnings


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def decimal_in_binary_array(decimal, array_length):
    a = [int(x) for x in bin(decimal)[2:]]
    a = [0]*(array_length-len(a)) + a
    return a


def find_neighbors(pos_index, m_b, n_b):
    pos_up = np.where(pos_index < n_b, pos_index + (m_b-1) * n_b, pos_index - n_b)

    # pos_down = pos_index + n_b
    pos_down = np.where(pos_index >= (m_b-1) * n_b, pos_index - (m_b-1) * n_b, pos_index + n_b)

    pos_left = np.where(pos_index % n_b == 0, pos_index - 1 + n_b, pos_index - 1)

    pos_right = np.where((pos_index + 1) % n_b == 0, pos_index + 1 - n_b, pos_index + 1)

    return pos_up, pos_down, pos_left, pos_right


def make_systematic(mat, is_pcm=False):
    r"""Bring binary matrix in its systematic form.

    Input
    -----
    mat : ndarray
        Binary matrix to be transformed to systematic form of shape `[k, n]`.

    is_pcm: bool
        Defaults to False. If true, ``mat`` is interpreted as parity-check
        matrix and, thus, the last k columns will be the identity part.

    Output
    ------
    mat_sys: ndarray
        Binary matrix in systematic form, i.e., the first `k` columns equal the
        identity matrix (or last `k` if ``is_pcm`` is True).

    column_swaps: list of int tuples
        A list of integer tuples that describes the swapped columns (in the
        order of execution).

    Note
    ----
    This algorithm (potentially) swaps columns of the input matrix. Thus, the
    resulting systematic matrix (potentially) relates to a permuted version of
    the code, this is defined by the returned list ``column_swap``.
    Note that, the inverse permutation must be applied in the inverse list
    order (in case specific columns are swapped multiple times).

    If a parity-check matrix is passed as input (i.e., ``is_pcm`` is True), the
    identity part will be re-arranged to the last columns."""

    m = mat.shape[0]
    n = mat.shape[1]

    assert m <= n, "Invalid matrix dimensions."

    # check for all-zero columns (=unchecked nodes)
    if is_pcm:
        c_node_deg = np.sum(mat, axis=0)
        if np.any(c_node_deg == 0):
            # warnings.warn("All-zero column in parity-check matrix detected. "
            #               "It seems as if the code contains unprotected nodes.")
            raise ValueError("All-zero column in parity-check matrix detected. ")

    mat = np.copy(mat)
    column_swaps = []  # store all column swaps

    # convert to bool for faster arithmetics
    mat = mat.astype(bool)

    # bring in upper triangular form
    for idx_c in range(m):
        success = False
        # step 1: find next leading "1"
        for idx_r in range(idx_c, m):
            # skip if entry is "0"
            if mat[idx_r, idx_c]:
                mat[[idx_c, idx_r]] = mat[[idx_r, idx_c]]  # swap rows
                success = True
                break

        # Could not find "1"-entry for column idx_c
        # => swap with columns from non-sys part
        # The task is to find a column with index idx_cc that has a "1" at
        # row idx_c
        if not success:
            for idx_cc in range(m, n):
                if mat[idx_c, idx_cc]:
                    # swap columns
                    mat[:, [idx_c, idx_cc]] = mat[:, [idx_cc, idx_c]]
                    column_swaps.append([idx_c, idx_cc])
                    success = True
                    break

        if not success:
            raise ValueError("Could not succeed; mat is not full rank?")

        # we can now assume a leading "1" at row idx_c
        for idx_r in range(idx_c + 1, m):
            if mat[idx_r, idx_c]:
                mat[idx_r, :] ^= mat[idx_c, :]  # bin. add of row idx_c to idx_r

    # remove upper triangle part in inverse order
    for idx_c in range(m - 1, -1, -1):
        for idx_r in range(idx_c - 1, -1, -1):
            if mat[idx_r, idx_c]:
                mat[idx_r, :] ^= mat[idx_c, :]  # bin. add of row idx_c to idx_r

    # verify results
    assert np.array_equal(mat[:, :m], np.eye(m)), \
        "Internal error, could not find systematic matrix."

    # bring identity part to end of matrix if parity-check matrix is provided
    if is_pcm:
        im = np.copy(mat[:, :m])
        mat[:, :m] = mat[:, -m:]
        mat[:, -m:] = im
        # and track column swaps
        for idx in range(m):
            column_swaps.append([idx, n - m + idx])

    # return integer array
    mat = mat.astype(int)
    return mat, column_swaps


def pcm2gm(pcm, verify_results=True):
    r"""Generate the generator matrix for a given parity-check matrix.

    This function brings ``pcm`` :math:`\mathbf{H}` in its systematic form and
    uses the following relation to find the generator matrix
    :math:`\mathbf{G}` in GF(2)

    .. math::

        \mathbf{G} = [\mathbf{I} |  \mathbf{M}]
        \Leftrightarrow \mathbf{H} = [\mathbf{M} ^t | \mathbf{I}]. \tag{1}

    This follows from the fact that for an all-zero syndrome, it must hold that

    .. math::

        \mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
        \mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}

    where :math:`\mathbf{c}` denotes an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    This leads to

    .. math::

     \mathbf{G} * \mathbf{H} ^t =: \mathbf{0}. \tag{2}

    It can be seen that (1) fulfills (2) as in GF(2) it holds that

    .. math::

        [\mathbf{I} |  \mathbf{M}] * [\mathbf{M} ^t | \mathbf{I}]^t
         = \mathbf{M} + \mathbf{M} = \mathbf{0}.

    Input
    -----
    pcm : ndarray
        Binary parity-check matrix of shape `[n-k, n]`.

    verify_results: bool
        Defaults to True. If True, it is verified that the generated
        generator matrix is orthogonal to the parity-check matrix in GF(2).

    Output
    ------
    : ndarray
        Binary generator matrix of shape `[k, n]`.

    Note
    ----
    This algorithm only works if ``pcm`` has full rank. Otherwise an error is
    raised.

    """
    n = pcm.shape[1]
    k = n - pcm.shape[0]

    assert k<n, "Invalid matrix dimensions."

    # bring pcm in systematic form
    pcm_sys, c_swaps = make_systematic(pcm, is_pcm=True)

    m_mat = np.transpose(np.copy(pcm_sys[:,:k]))
    i_mat = np.eye(k)
    gm = np.concatenate((i_mat, m_mat), axis=1)

    # undo column swaps
    for l in c_swaps[::-1]: # reverse ordering when going through list
        gm[:,[l[0], l[1]]] = gm[:,[l[1], l[0]]] # swap columns

    if verify_results:
        assert verify_gm_pcm(gm=gm, pcm=pcm), \
            "Resulting parity-check matrix does not match to generator matrix."
    return gm


def verify_gm_pcm(gm, pcm):
    r"""Verify that generator matrix :math:`\mathbf{G}` ``gm`` and parity-check
    matrix :math:`\mathbf{H}` ``pcm`` are orthogonal in GF(2).

    For an all-zero syndrome, it must hold that

    .. math::

        \mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
        \mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}

    where :math:`\mathbf{c}` denotes an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    As :math:`\mathbf{u}` can be arbitrary it follows that

    .. math::
        \mathbf{H} * \mathbf{G} ^t =: \mathbf{0}.

    Input
    -----
    gm : ndarray
        Binary generator matrix of shape `[k, n]`.

    pcm : ndarray
        Binary parity-check matrix of shape `[n-k, n]`.

    Output
    ------
    : bool
        True if ``gm`` and ``pcm`` define a valid pair of parity-check and
        generator matrices in GF(2).
    """

    # check for valid dimensions
    k = gm.shape[0]
    n = gm.shape[1]

    n_pcm = pcm.shape[1]
    k_pcm = n_pcm - pcm.shape[0]

    assert k==k_pcm, "Inconsistent shape of gm and pcm."
    assert n==n_pcm, "Inconsistent shape of gm and pcm."

    # check that both matrices are binary
    assert ((gm==0) | (gm==1)).all(), "gm is not binary."
    assert ((pcm==0) | (pcm==1)).all(), "pcm is not binary."

    # check for zero syndrome
    s = np.mod(np.matmul(pcm, np.transpose(gm)), 2) # mod2 to account for GF(2)
    return np.sum(s)==0 # Check for Non-zero syndrom of H*G'


class LinearEncoder(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""LinearEncoder(enc_mat, is_pcm=False, dtype=tf.float32, **kwargs)

    Linear binary encoder for a given generator or parity-check matrix ``enc_mat``.

    If ``is_pcm`` is True, ``enc_mat`` is interpreted as parity-check
    matrix and internally converted to a corresponding generator matrix.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
    enc_mat : [k, n] or [n-k, n], ndarray
        Binary generator matrix of shape `[k, n]`. If ``is_pcm`` is
        True, ``enc_mat`` is interpreted as parity-check matrix of shape
        `[n-k, n]`.

    dtype: tf.DType
        Defaults to `tf.float32`. Defines the datatype for the output dtype.

    Input
    -----
    inputs: [...,k], tf.float32
        2+D tensor containing information bits.

    Output
    ------
    : [...,n], tf.float32
        2+D tensor containing codewords with same shape as inputs, except the
        last dimension changes to `[...,n]`.

    Raises
    ------
    AssertionError
        If the encoding matrix is not a valid binary 2-D matrix.

    Note
    ----
        If ``is_pcm`` is True, this layer uses
        :class:`~sionna.fec.utils.pcm2gm` to find the generator matrix for
        encoding. Please note that this imposes a few constraints on the
        provided parity-check matrix such as full rank and it must be binary.

        Note that this encoder is generic for all binary linear block codes
        and, thus, cannot implement any code specific optimizations. As a
        result, the encoding complexity is :math:`O(k^2)`. Please consider code
        specific encoders such as the
        :class:`~sionna.fec.polar.encoding.Polar5GEncoder` or
        :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder` for an improved
        encoding performance.
    """

    def __init__(self,
                 enc_mat,
                 is_pcm=False,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        # tf.int8 currently not supported by tf.matmult
        assert (dtype in
               (tf.float16, tf.float32, tf.float64, tf.int32, tf.int64)), \
               "Unsupported dtype."

        # check input values for consistency
        assert isinstance(is_pcm, bool), \
                                    'is_parity_check must be bool.'

        # verify that enc_mat is binary
        assert ((enc_mat==0) | (enc_mat==1)).all(), "enc_mat is not binary."
        assert (len(enc_mat.shape)==2), "enc_mat must be 2-D array."

        # in case parity-check matrix is provided, convert to generator matrix
        if is_pcm:
            self._gm = pcm2gm(enc_mat, verify_results=True)
        else:
            self._gm = enc_mat

        self._k = self._gm.shape[0]
        self._n = self._gm.shape[1]
        self._coderate = self._k / self._n

        assert (self._k<=self._n), "Invalid matrix dimensions."

        self._gm = tf.cast(self._gm, dtype=self.dtype)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits per codeword."""
        return self._k

    @property
    def n(self):
        "Codeword length."
        return self._n

    @property
    def gm(self):
        "Generator matrix used for encoding."
        return self._gm

    @property
    def coderate(self):
        """Coderate of the code."""
        return self._coderate

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build, but check for valid shapes."""
        assert input_shape[-1]==self._k, "Invalid input shape."
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

    def call(self, inputs):
        """Generic encoding function based on generator matrix multiplication.
        """

        c = tf.linalg.matmul(inputs, self._gm)

        # faster implementation of tf.math.mod(c, 2)
        c_uint8 = tf.cast(c, tf.uint8)
        c_bin = tf.bitwise.bitwise_and(c_uint8, tf.constant(1, tf.uint8))
        c = tf.cast(c_bin, self.dtype)

        return c


class BinarySource(Layer):
    """BinarySource(dtype=tf.float32, seed=None, **kwargs)

    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : tf.DType
        Defines the output datatype of the layer.
        Defaults to `tf.float32`.

    seed : int or None
        Set the seed for the random generator used to generate the bits.
        Set to `None` for random initialization of the RNG.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor filled with random binary values.
    """
    def __init__(self, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = tf.random.Generator.from_seed(self._seed)

    def call(self, inputs):
        if self._seed is not None:
            return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)
        else:
            return tf.cast(tf.random.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)


def ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid=None):
    r"""Compute the noise variance `No` for a given `Eb/No` in dB.

    The function takes into account the number of coded bits per constellation
    symbol, the coderate, as well as possible additional overheads related to
    OFDM transmissions, such as the cyclic prefix and pilots.

    The value of `No` is computed according to the following expression

    .. math::
        N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}

    where :math:`2^M` is the constellation size, i.e., :math:`M` is the
    average number of coded bits per constellation symbol,
    :math:`E_s=1` is the average energy per constellation per symbol,
    :math:`r\in(0,1]` is the coderate,
    :math:`E_b` is the energy per information bit,
    and :math:`N_o` is the noise power spectral density.
    For OFDM transmissions, :math:`E_s` is scaled
    according to the ratio between the total number of resource elements in
    a resource grid with non-zero energy and the number
    of resource elements used for data transmission. Also the additionally
    transmitted energy during the cyclic prefix is taken into account, as
    well as the number of transmitted streams per transmitter.

    Input
    -----
    ebno_db : float
        The `Eb/No` value in dB.

    num_bits_per_symbol : int
        The number of bits per symbol.

    coderate : float
        The coderate used.

    resource_grid : ResourceGrid
        An (optional) instance of :class:`~sionna.ofdm.ResourceGrid`
        for OFDM transmissions.

    Output
    ------
    : float
        The value of :math:`N_o` in linear scale.
    """

    if tf.is_tensor(ebno_db):
        dtype = ebno_db.dtype
    else:
        dtype = tf.float32

    ebno = tf.math.pow(tf.cast(10., dtype), ebno_db/10.)

    energy_per_symbol = 1
    if resource_grid is not None:
        # Divide energy per symbol by the number of transmitted streams
        energy_per_symbol /= resource_grid.num_streams_per_tx

        # Number of nonzero energy symbols.
        # We do not account for the nulled DC and guard carriers.
        cp_overhead = resource_grid.cyclic_prefix_length \
                      / resource_grid.fft_size
        num_syms = resource_grid.num_ofdm_symbols * (1 + cp_overhead) \
                    * resource_grid.num_effective_subcarriers
        energy_per_symbol *= num_syms / resource_grid.num_data_symbols

    no = 1/(ebno * coderate * tf.cast(num_bits_per_symbol, dtype) \
          / tf.cast(energy_per_symbol, dtype))

    return no


def compute_ber(b, b_hat):
    """Computes the bit error rate (BER) between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BER.
    """
    ber = tf.not_equal(b, b_hat)
    ber = tf.cast(ber, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ber)


def compute_bler(b, b_hat):
    """Computes the block error rate (BLER) between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BLER.
    """
    bler = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
    bler = tf.cast(bler, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(bler)
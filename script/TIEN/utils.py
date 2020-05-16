import tensorflow as tf
from numpy.core import multiarray
from numpy.core.multiarray import empty
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import _Linear


def prelu(_x, scope=''):
  """parametric ReLU activation"""
  with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
    _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                             dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


class QAAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(QAAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, att_score):
    return self.call(inputs, state, att_score)

  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
          [inputs, state],
          2 * self._num_units,
          True,
          bias_initializer=bias_ones,
          kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
          [inputs, r_state],
          self._num_units,
          True,
          bias_initializer=self._bias_initializer,
          kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    new_h = (1. - att_score) * state + att_score * c
    return new_h, new_h


class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, att_score):
    return self.call(inputs, state, att_score)

  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
          [inputs, state],
          2 * self._num_units,
          True,
          bias_initializer=bias_ones,
          kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
          [inputs, r_state],
          self._num_units,
          True,
          bias_initializer=self._bias_initializer,
          kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h


def prelu(_x, scope=''):
  """parametric ReLU activation"""
  with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
    _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                             dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
  if isinstance(facts, tuple):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    facts = tf.concat(facts, 2)
    print ("querry_size mismatch")
    query = tf.concat(values=[
      query,
      query,
    ], axis=1)

  if time_major:
    # (T,B,D) => (B,T,D)
    facts = tf.array_ops.transpose(facts, [1, 0, 2])
  mask = tf.equal(mask, tf.ones_like(mask))
  facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
  querry_size = query.get_shape().as_list()[-1]
  queries = tf.tile(query, [1, tf.shape(facts)[1]])
  queries = tf.reshape(queries, tf.shape(facts))
  din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
  scores = d_layer_3_all
  # Mask
  # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
  key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
  paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
  scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

  # Scale
  # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

  # Activation
  if softmax_stag:
    scores = tf.nn.softmax(scores)  # [B, 1, T]

  # Weighted sum
  if mode == 'SUM':
    output = tf.matmul(scores, facts)  # [B, 1, H]
    # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
  else:
    scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
    output = facts * tf.expand_dims(scores, -1)
    output = tf.reshape(output, tf.shape(facts))
  return output


# https://github.com/mouna99/dien/blob/master/script/utils.py
def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False, scope="user_hist_group"):
  with tf.variable_scope(scope):
    if isinstance(facts, tuple):
      # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
      facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
      facts = tf.expand_dims(facts, 1)

    if time_major:
      # (T,B,D) => (B,T,D)
      facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
      scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
      scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
      output = tf.matmul(scores, facts)  # [B, 1, H]
      # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
      scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
      output = facts * tf.expand_dims(scores, -1)
      output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
      return output, scores
    return output


# https://github.com/jinze1994/ATRank/blob/master/atrank/model.py
def attention_net(enc, sl, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse):
  with tf.variable_scope("all", reuse=reuse):
    with tf.variable_scope("user_hist_group"):
      for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ### Multihead Attention
          enc, stt_vec = multihead_attention(queries=enc,
                                             queries_length=sl,
                                             keys=enc,
                                             keys_length=sl,
                                             num_units=num_units,
                                             num_heads=num_heads,
                                             dropout_rate=dropout_rate,
                                             is_training=is_training,
                                             scope="self_attention"
                                             )

          ### Feed Forward
          enc = feedforward(enc,
                            num_units=[num_units // 4, num_units],
                            scope="feed_forward", reuse=reuse)

    dec = tf.expand_dims(dec, 1)
    with tf.variable_scope("item_feature_group"):
      for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ## Multihead Attention ( vanilla attention)
          dec, att_vec = multihead_attention(queries=dec,
                                             queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32),
                                             keys=enc,
                                             keys_length=sl,
                                             num_units=num_units,
                                             num_heads=num_heads,
                                             dropout_rate=dropout_rate,
                                             is_training=is_training,
                                             scope="vanilla_attention")

          ## Feed Forward
          dec = feedforward(dec,
                            num_units=[num_units // 4, num_units],
                            scope="feed_forward", reuse=reuse)

    dec = tf.reshape(dec, [-1, num_units])
    return dec, att_vec, stt_vec


def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
  '''Applies multihead attention.
  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs)  # (N, T_q, C)

  return outputs, att_vec


def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                reuse=None):
  '''Point-wise feed forward net.
  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
              "activation": tf.nn.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
              "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = normalize(outputs)

  return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
  '''Applies layer normalization.
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
    `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

  return outputs


# https://github.com/ouououououou/DIB-PEB-Sequential-RS/blob/master/recommender/RUM_Ksoft_mulcha.py
def dynamic_item_block_user(user_embedding, user_memory_embedding, numFactor=32):
  """user_embedding shape: (train_batch, numFactor)
     user_memory_embedding shape: (train_batch * input_size * familiar_user, numFactor)"""
  user_embedding = tf.reshape(user_embedding, [-1, 1, numFactor])
  user_memory_embedding = tf.reshape(user_memory_embedding,
                                     [-1, tf.shape(user_memory_embedding)[1], numFactor])

  weight = tf.reshape(tf.div(tf.matmul(user_memory_embedding, user_embedding, transpose_b=True),
                             tf.sqrt(tf.to_float(numFactor))),
                      [-1, 1, tf.shape(user_memory_embedding)[1]])

  # attention = tf.expand_dims(tf.nn.softmax(weight, axis=2), axis=3)
  attention = tf.expand_dims(tf.nn.softmax(weight, dim=2), axis=3)
  out = tf.reduce_mean(tf.multiply(
    tf.reshape(user_memory_embedding, [-1, 1, tf.shape(user_memory_embedding)[1], numFactor]),
    attention), axis=2)
  "return shape: (train_batch, input_size, numFactor)"
  return out

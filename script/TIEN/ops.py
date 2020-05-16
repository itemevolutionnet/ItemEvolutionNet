import tensorflow as tf


def attention_net_v1(enc, sl, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, scope='all',
                     value=None):
  with tf.variable_scope(scope, reuse=reuse):
    dec = tf.expand_dims(dec, 1)
    with tf.variable_scope("item_feature_group"):
      for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ## Multihead Attention ( vanilla attention)
          dec, att_vec = multihead_attention_v1(queries=dec,
                                                queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32),
                                                keys=enc,
                                                keys_length=sl,
                                                num_units=num_units,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                is_training=is_training,
                                                scope="vanilla_attention",
                                                value=value)

          ## Feed Forward
          dec = feedforward_v1(dec, num_units=[num_units // 2, num_units],
                               scope="feed_forward", reuse=reuse)
    dec = tf.reshape(dec, [-1, num_units])
    return dec, att_vec


def multihead_attention_v1(queries,
                           queries_length,
                           keys,
                           keys_length, \
                           num_units=None,
                           num_heads=8,
                           dropout_rate=0,
                           is_training=True,
                           scope="multihead_attention",
                           reuse=None,
                           first_n_att_weight_report=20,
                           value=None):
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
    Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
    if value is None:
      V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
    else:
      V = tf.layers.dense(value, num_units, activation=None)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    from tensorflow.contrib import layers
    Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
    K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    # outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

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
    # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    ########################
    ########################
    # summary
    keys_masks_tmp = tf.reshape(tf.cast(key_masks, tf.float32), [-1, tf.shape(keys)[1]])
    defined_length = tf.constant(first_n_att_weight_report, dtype=tf.float32, name="%s_defined_length" % (scope))
    greater_than_define = tf.cast(tf.greater(tf.reduce_sum(keys_masks_tmp, axis=1), defined_length), tf.float32)
    greater_than_define_exp = tf.tile(tf.expand_dims(greater_than_define, -1), [1, tf.shape(keys)[1]])

    weight = tf.reshape(outputs, [-1, tf.shape(keys)[1]]) * greater_than_define_exp
    weight_map = tf.reshape(weight, [-1, tf.shape(queries)[1], tf.shape(keys)[1]])  # BxL1xL2
    greater_than_define_exp_map = tf.reshape(greater_than_define_exp,
                                             [-1, tf.shape(queries)[1], tf.shape(keys)[1]])  # BxL1xL2
    weight_map_mean = tf.reduce_sum(weight_map, 0) / (
        tf.reduce_sum(greater_than_define_exp_map, axis=0) + 1e-5)  # L1xL2
    report_image = tf.expand_dims(tf.expand_dims(weight_map_mean, -1), 0)  # 1xL1xL2x1
    tf.summary.image("%s_attention" % (scope),
                     report_image[:, :first_n_att_weight_report, :first_n_att_weight_report,
                     :])  # 1x10x10x1
    ########################
    ########################

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    # outputs += queries

    # Normalize
    # outputs = normalize(outputs)  # (N, T_q, C)

  return outputs, att_vec


def feedforward_v1(inputs,
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
    # outputs = normalize(outputs)

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


def near_k_behaviors(sequence, seq_len, k=5):
  reverse_seq = tf.reverse_sequence(sequence, seq_len, 1, 0)
  reverse_seq_k = reverse_seq[:, :k, :]
  seq_len_k = tf.clip_by_value(seq_len, 0, k)
  sequence_k = tf.reverse_sequence(reverse_seq_k,
                                   seq_len_k,
                                   1,
                                   0)

  mask = tf.sequence_mask(seq_len_k, tf.shape(sequence_k)[1],
                          dtype=tf.float32)  # [B, T]
  mask = tf.expand_dims(mask, -1)
  return sequence_k, seq_len_k, mask

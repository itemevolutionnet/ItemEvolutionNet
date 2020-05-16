import ops
from rnn import dynamic_rnn
from utils import *


class Model(object):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM):
    with tf.name_scope('Inputs'):
      self.item_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
      self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
      self.user_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='user_his_batch_ph')

      self.item_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
      self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
      self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')

      self.user_bh_seq_len_ph = tf.placeholder(tf.int32, [None], name='user_bh_seq_len_ph')
      self.item_bh_seq_len_ph = tf.placeholder(tf.int32, [None], name='item_bh_seq_len_ph')

      self.user_bh_time_batch_ph = tf.placeholder(tf.int32, [None, None])
      self.item_bh_time_batch_ph = tf.placeholder(tf.int32, [None, None])

      self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
      self.lr = tf.placeholder(tf.float64, [])

    # Embedding layer
    with tf.name_scope('Embedding_layer'):
      self.item_embeddings_var = tf.get_variable("item_embedding_var", [n_mid, EMBEDDING_DIM])
      self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
      self.user_embeddings_var = tf.get_variable("user_embedding_var", [n_uid, EMBEDDING_DIM * 2])

      tf.summary.histogram('mid_embeddings_var', self.item_embeddings_var)
      tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
      tf.summary.histogram('user_embeddings_var', self.user_embeddings_var)

      self.item_batch_embedded = tf.nn.embedding_lookup(self.item_embeddings_var, self.item_batch_ph)
      self.item_his_batch_embedded = tf.nn.embedding_lookup(self.item_embeddings_var, self.item_his_batch_ph)

      self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
      self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)

      self.user_embedded = tf.nn.embedding_lookup(self.user_embeddings_var, self.uid_batch_ph)
      self.item_bh_embedded = tf.nn.embedding_lookup(self.user_embeddings_var, self.user_his_batch_ph)

    self.item_embedded = tf.concat([self.item_batch_embedded, self.cat_batch_embedded], 1)
    self.user_bh_embedded_tmp = tf.concat([self.item_his_batch_embedded, self.cat_his_batch_embedded], 2)

    user_bh_time_embedded = tf.one_hot(self.user_bh_time_batch_ph, 12, dtype=tf.float32)
    user_bh_embedded = tf.concat([self.user_bh_embedded_tmp, user_bh_time_embedded], -1)
    self.user_bh_embedded = tf.layers.dense(user_bh_embedded, EMBEDDING_DIM * 2, name='user_bh_time_emb')

    # mask
    self.user_bh_mask = tf.sequence_mask(self.user_bh_seq_len_ph, tf.shape(self.user_bh_embedded)[1],
                                         dtype=tf.float32)
    self.item_bh_mask = tf.sequence_mask(self.item_bh_seq_len_ph, tf.shape(self.item_bh_embedded)[1],
                                         dtype=tf.float32)
    self.user_bh_mask_d = tf.expand_dims(self.user_bh_mask, -1)
    self.item_bh_mask_d = tf.expand_dims(self.item_bh_mask, -1)

    self.user_bh_masked_embedded = self.user_bh_embedded_tmp * self.user_bh_mask_d
    self.user_bh_embedded_sum = tf.reduce_sum(self.user_bh_masked_embedded, 1)

    self.global_step = tf.Variable(0, trainable=False, name='global_step')

  def build_fc_net(self, inp):
    fc1 = tf.layers.dense(inp, 256, activation=tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
    fc3 = tf.layers.dense(fc2, 1, activation=None, name='fc3')
    self.y_hat = tf.nn.sigmoid(fc3)

  def build_fc_net_hyper(self, inp, hidden_units):
    fc = inp
    if len(hidden_units) > 1:
      for i in xrange(len(hidden_units) - 1):
        fc = tf.layers.dense(fc, hidden_units[i], activation=tf.nn.relu, name='fc{}'.format(i))
    fc3 = tf.layers.dense(fc, hidden_units[-1], activation=None, name='fcn')
    self.y_hat = tf.nn.sigmoid(fc3)

  def build_loss(self):
    # loss
    self.log_loss = tf.losses.log_loss(self.target_ph, self.y_hat)
    self.loss = self.log_loss
    tf.summary.scalar('loss', self.loss)
    # optimizer and training step
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, self.global_step)

    # Accuracy metric
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
    tf.summary.scalar('accuracy', self.accuracy)

    self.merged = tf.summary.merge_all()

  def train(self, sess, inps):
    probs, loss, accuracy, _, summary = sess.run([self.y_hat, self.loss, self.accuracy, self.optimizer, self.merged],
                                                 feed_dict={
                                                   self.uid_batch_ph: inps[0],
                                                   self.item_batch_ph: inps[1],
                                                   self.cat_batch_ph: inps[2],
                                                   self.item_his_batch_ph: inps[3],
                                                   self.cat_his_batch_ph: inps[4],
                                                   self.target_ph: inps[6],
                                                   self.user_bh_seq_len_ph: inps[7],
                                                   self.lr: inps[8],
                                                   self.user_his_batch_ph: inps[9],
                                                   self.item_bh_seq_len_ph: inps[10],
                                                   self.item_bh_mask: inps[11],
                                                   self.item_bh_time_batch_ph: inps[12],
                                                   self.user_bh_time_batch_ph: inps[13]
                                                 })
    return probs, loss, accuracy, summary

  def calculate(self, sess, inps):
    probs, loss, accuracy, summary = sess.run([self.y_hat, self.loss, self.accuracy, self.merged], feed_dict={
      self.uid_batch_ph: inps[0],
      self.item_batch_ph: inps[1],
      self.cat_batch_ph: inps[2],
      self.item_his_batch_ph: inps[3],
      self.cat_his_batch_ph: inps[4],
      self.target_ph: inps[6],
      self.user_bh_seq_len_ph: inps[7],
      self.user_his_batch_ph: inps[8],
      self.item_bh_seq_len_ph: inps[9],
      self.item_bh_time_batch_ph: inps[11],
      self.user_bh_time_batch_ph: inps[12]
    })
    return probs, loss, accuracy, summary

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path + '/model.ckpt', global_step=self.global_step.eval())

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path)


class DNN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM):
    super(DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum], 1)
    self.build_fc_net(inp)
    self.build_loss()


class PNN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM):
    super(PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum], 1)
    self.build_fc_net(inp)
    self.build_loss()


class SVDPP(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum, item_fnum):
    super(SVDPP, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('user_feature_rep'):
      self.user_feat_w_list = []
      for i in range(user_fnum):
        self.user_feat_w_list.append(
          tf.get_variable('user_feat_w_%d' % i, [], initializer=tf.truncated_normal_initializer))
      self.user_eb_rep = self.user_embedded[:, :EMBEDDING_DIM] * self.user_feat_w_list[0]
      for i in range(1, user_fnum):
        self.user_eb_rep += self.user_embedded[:, i * EMBEDDING_DIM:(i + 1) * EMBEDDING_DIM] * self.user_feat_w_list[i]

    with tf.name_scope('item_feature_rep'):
      self.item_feat_w_list = []
      for i in range(item_fnum):
        self.item_feat_w_list.append(
          tf.get_variable('item_feat_w_%d' % i, [], initializer=tf.truncated_normal_initializer))
      self.item_eb_rep = self.item_embedded[:, :EMBEDDING_DIM] * self.item_feat_w_list[0]
      self.user_seq_rep = self.user_bh_embedded_tmp[:, :, :EMBEDDING_DIM] * self.item_feat_w_list[0]
      for i in range(1, item_fnum):
        self.item_eb_rep += self.item_embedded[:, i * EMBEDDING_DIM:(i + 1) * EMBEDDING_DIM] * self.item_feat_w_list[i]
        self.user_seq_rep += self.user_bh_embedded_tmp[:, :, i * EMBEDDING_DIM:(i + 1) * EMBEDDING_DIM] * \
                             self.item_feat_w_list[
                               i]

    self.user_seq_mask = tf.expand_dims(
      tf.sequence_mask(self.user_bh_seq_len_ph, tf.shape(self.item_his_batch_ph)[1], dtype=tf.float32), 2)
    self.user_seq_rep = self.user_seq_rep * self.user_seq_mask
    self.neighbor = tf.reduce_sum(self.user_seq_rep, axis=1)
    self.norm_neighbor = self.neighbor / tf.sqrt(tf.expand_dims(tf.norm(self.user_seq_rep, 1, (1, 2)), 1))
    self.latent_score = tf.reduce_sum(self.item_eb_rep * (self.user_eb_rep + self.norm_neighbor), 1)
    self.y_hat = tf.nn.sigmoid(self.latent_score)
    self.y_hat = tf.reshape(self.y_hat, [-1, 1])
    self.build_loss()


class GRU4REC(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(GRU4REC, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru1")
      _, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru2")

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)

    self.build_fc_net(inp)
    self.build_loss()


class UBGRUA(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(UBGRUA, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('rnn'):
      gru_outputs_u, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                                       sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                                       scope="gru")
      final_state_u, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                              num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                              is_training=False, reuse=False)
    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)
    self.build_fc_net(inp)
    self.build_loss()


class DIN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    # Attention layer
    with tf.name_scope('Attention_layer'):
      attention_output = din_attention(self.item_embedded, self.user_bh_embedded, HIDDEN_SIZE, self.user_bh_mask)
      final_state_u = tf.reduce_sum(attention_output, 1)

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)

    self.build_fc_net(inp)
    self.build_loss()


class ATRANK(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(ATRANK, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    final_state_u, self.att, self.stt = attention_net(
      self.user_bh_embedded,
      self.user_bh_seq_len_ph,
      self.item_embedded,
      HIDDEN_SIZE,
      num_heads=4,
      num_blocks=1,
      dropout_rate=0.,
      is_training=False,
      reuse=False)

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)
    self.build_fc_net(inp)
    self.build_loss()


class CASER(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum, item_fnum, max_len=100):
    super(CASER, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('user_seq_cnn'):
      self.user_bh_embedded = tf.reshape(self.user_bh_embedded, [-1, max_len, EMBEDDING_DIM * item_fnum])
      # horizontal filters
      filters_user = 1
      h_kernel_size_user = [50, EMBEDDING_DIM * item_fnum]
      v_kernel_size_user = [self.user_bh_embedded.get_shape().as_list()[1], 1]

      self.user_bh_embedded = tf.expand_dims(self.user_bh_embedded, 3)
      conv1 = tf.layers.conv2d(self.user_bh_embedded, filters_user, h_kernel_size_user)
      max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
      user_hori_out = tf.reshape(max1, [-1, filters_user])  # [B, F]

      # vertical
      conv2 = tf.layers.conv2d(self.user_bh_embedded, filters_user, v_kernel_size_user)
      conv2 = tf.reshape(conv2, [-1, EMBEDDING_DIM * item_fnum, filters_user])
      user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, EMBEDDING_DIM * item_fnum])

      inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                       self.item_embedded * self.user_bh_embedded_sum, user_hori_out, user_vert_out],
                      1)

    # fully connected layer
    self.build_fc_net(inp)
    self.build_loss()


class DIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    # RNN layer(-s)
    with tf.name_scope('rnn_1'):
      gru_outputs_u, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                     sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                     scope="gru1")
    # Attention layer
    with tf.name_scope('Attention_layer_1'):
      att_outputs, alphas = din_fcn_attention(self.item_embedded, gru_outputs_u, HIDDEN_SIZE, self.user_bh_mask,
                                              softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
    with tf.name_scope('rnn_2'):
      _, final_state_u = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                     att_scores=tf.expand_dims(alphas, -1),
                                     sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                     scope="gru2")
    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TopoLSTM(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(TopoLSTM, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru1")
      _, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru2")

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    with tf.name_scope('rnn_1'):
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t, dtype=tf.float32,
                                           scope="gru3")
      _, final_state_i = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_i,
                                           sequence_length=self.item_bh_seq_len_t, dtype=tf.float32,
                                           scope="gru4")

    ###########################

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u,
                     self.item_bh_embedded_sum, final_state_i], 1)
    self.build_fc_net(inp)
    self.build_loss()


class DIB(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(DIB, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru1")
      _, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru2")

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)

    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    with tf.name_scope('dib'):
      # user attention
      final_state_i = tf.reshape(
        dynamic_item_block_user(self.user_embedded, self.item_bh_t * self.item_bh_mask_t, numFactor=HIDDEN_SIZE),
        shape=[-1, HIDDEN_SIZE])

    ###########################

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u,
                     self.item_bh_embedded_sum, final_state_i], 1)
    self.build_fc_net(inp)
    self.build_loss()


class IBGRUA(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(IBGRUA, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru1")
      _, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru2")

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)

    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    with tf.name_scope('dib'):
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t, dtype=tf.float32,
                                           scope="gru")
      # user attention
      final_state_i = tf.reshape(
        dynamic_item_block_user(self.user_embedded, gru_outputs_i * self.item_bh_mask_t, numFactor=HIDDEN_SIZE),
        shape=[-1, HIDDEN_SIZE])

    # with tf.name_scope('rnn_1'):
    #   gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t,
    #                                        sequence_length=self.item_bh_seq_len_t, dtype=tf.float32,
    #                                        scope="gru")
    #   final_state_i, _ = ops.attention_net_v1(gru_outputs_i, sl=self.item_bh_seq_len_t, dec=self.user_embedded,
    #                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
    #                                           is_training=False, reuse=False)

    inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
                     self.item_embedded * self.user_bh_embedded_sum, final_state_u,
                     self.item_bh_embedded_sum, final_state_i], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TIEN_sumagg(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(TIEN_sumagg, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru")
      final_state_u, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                              num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                              is_training=False, reuse=False)

    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, final_state_u, self.item_bh_embedded_sum], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TIEN_timeatt(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(TIEN_timeatt, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

    # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      dec = self.user_embedded
      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t,
                                      dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

      inp = tf.concat(
        [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
         self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum, u_att, i_att], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TIEN_robust(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(TIEN_robust, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

    # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

      inp = tf.concat(
        [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
         self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
         u_att, i_att], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TIEN_timeaware(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T):
    super(TIEN_timeaware, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state

    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       u_att, i_att,
       i_att_ta], 1)
    self.build_fc_net(inp)
    self.build_loss()


class TIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T, HIDDEN_UNITS):
    super(TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state

    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       u_att, i_att,
       i_att_ta], 1)
    self.build_fc_net_hyper(inp, HIDDEN_UNITS)
    self.build_loss()


class ATRANK_TIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(ATRANK_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    final_state_u, self.att, self.stt = attention_net(
      self.user_bh_embedded,
      self.user_bh_seq_len_ph,
      self.item_embedded,
      HIDDEN_SIZE,
      num_heads=4,
      num_blocks=1,
      dropout_rate=0.,
      is_training=False,
      reuse=False)

    # inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
    #                  self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)

    SEQ_USER_T = 50
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state
    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       final_state_u, i_att,
       i_att_ta], 1)
    self.build_fc_net(inp)
    self.build_loss()


class GRU4REC_TIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(GRU4REC_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru1")
      _, final_state_u = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                           sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                           scope="gru2")

    # inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
    #                  self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)
    SEQ_USER_T = 50
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state
    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       final_state_u, i_att,
       i_att_ta], 1)
    self.build_fc_net(inp)
    self.build_loss()


class CASER_TIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum, item_fnum, max_len=100):
    super(CASER_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    with tf.name_scope('user_seq_cnn'):
      self.user_bh_embedded_reshape = tf.reshape(self.user_bh_embedded, [-1, max_len, EMBEDDING_DIM * item_fnum])
      # horizontal filters
      filters_user = 1
      h_kernel_size_user = [50, EMBEDDING_DIM * item_fnum]
      v_kernel_size_user = [self.user_bh_embedded_reshape.get_shape().as_list()[1], 1]

      self.user_bh_embedded_reshape = tf.expand_dims(self.user_bh_embedded_reshape, 3)
      conv1 = tf.layers.conv2d(self.user_bh_embedded_reshape, filters_user, h_kernel_size_user)
      max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
      user_hori_out = tf.reshape(max1, [-1, filters_user])  # [B, F]

      # vertical
      conv2 = tf.layers.conv2d(self.user_bh_embedded_reshape, filters_user, v_kernel_size_user)
      conv2 = tf.reshape(conv2, [-1, EMBEDDING_DIM * item_fnum, filters_user])
      user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, EMBEDDING_DIM * item_fnum])

      # inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
      #                  self.item_embedded * self.user_bh_embedded_sum, user_hori_out, user_vert_out],
      #                 1)

    SEQ_USER_T = 50
    HIDDEN_SIZE = EMBEDDING_DIM * 2
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state
    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       user_hori_out, user_vert_out, i_att,
       i_att_ta], 1)

    # fully connected layer
    self.build_fc_net(inp)
    self.build_loss()


class DIEN_TIEN(Model):
  def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE):
    super(DIEN_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

    # RNN layer(-s)
    with tf.name_scope('rnn_1'):
      gru_outputs_u, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.user_bh_embedded,
                                     sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                     scope="gru1")
    # Attention layer
    with tf.name_scope('Attention_layer_1'):
      att_outputs, alphas = din_fcn_attention(self.item_embedded, gru_outputs_u, HIDDEN_SIZE, self.user_bh_mask,
                                              softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
    with tf.name_scope('rnn_2'):
      _, final_state_u = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=gru_outputs_u,
                                     att_scores=tf.expand_dims(alphas, -1),
                                     sequence_length=self.user_bh_seq_len_ph, dtype=tf.float32,
                                     scope="gru2")
    # inp = tf.concat([self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
    #                  self.item_embedded * self.user_bh_embedded_sum, final_state_u], 1)

    SEQ_USER_T = 50
    print "item baheviors length %d" % SEQ_USER_T

    ###########################
    # near k behavior sum-pooling, the best T=5
    self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=5)
    self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
    self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
    ###########################

    ###########################
    # near t behavior
    self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = ops.near_k_behaviors(self.item_bh_embedded,
                                                                                       self.item_bh_seq_len_ph,
                                                                                       k=SEQ_USER_T)
    item_bh_time_embedded = tf.one_hot(self.item_bh_time_batch_ph, 12, dtype=tf.float32)
    self.item_bh_time_embeeded_t, _, _ = ops.near_k_behaviors(item_bh_time_embedded, self.item_bh_seq_len_ph,
                                                              k=SEQ_USER_T)
    ###########################

    # 2.sequential modeling for user/item behaviors
    with tf.name_scope('rnn'):
      gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.user_bh_embedded,
                                           sequence_length=self.user_bh_seq_len_ph,
                                           dtype=tf.float32,
                                           scope="gru_ub")
      gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                           inputs=self.item_bh_t,
                                           sequence_length=self.item_bh_seq_len_t,
                                           dtype=tf.float32,
                                           scope="gru_ib")

    # 3.time-signal
    with tf.name_scope('time-signal'):
      item_bh_time_embeeded = tf.layers.dense(self.item_bh_time_embeeded_t, HIDDEN_SIZE, activation=None,
                                              name='item_bh_time_emb')

      # 4. attention layer
    with tf.name_scope('att'):
      gru_outputs_ib_with_t = gru_outputs_i + item_bh_time_embeeded
      u_att, _ = ops.attention_net_v1(gru_outputs_u, sl=self.user_bh_seq_len_ph, dec=self.item_embedded,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
      # 5. prevent noisy
      self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t
      self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
      dec = self.user_embedded + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32),
                                                                          [-1, 1]) + 1e5

      i_att, _ = ops.attention_net_v1(gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec,
                                      num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                      is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

    # 5.time-aware representation layer
    with tf.name_scope('time_gru'):
      _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                      inputs=item_bh_time_embeeded,
                                      sequence_length=self.item_bh_seq_len_t,
                                      dtype=tf.float32,
                                      scope="gru_it")
      i_att_ta = i_att + it_state

    inp = tf.concat(
      [self.user_embedded, self.item_embedded, self.user_bh_embedded_sum,
       self.item_embedded * self.user_bh_embedded_sum, self.item_bh_embedded_sum,
       final_state_u, i_att,
       i_att_ta], 1)
    self.build_fc_net(inp)
    self.build_loss()

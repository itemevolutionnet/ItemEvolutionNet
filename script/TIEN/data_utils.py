import numpy


def prepare_data(input, target, maxlen=None, seq_user_maxlen=None):
  # x: a list of sentences
  lengths_mid = [len(s[4]) for s in input]
  seqs_mid = [inp[3] for inp in input]
  seqs_cat = [inp[4] for inp in input]
  seqs_user = [inp[5] for inp in input]
  seqs_user_t = [inp[6] for inp in input]
  seqs_mid_t = [inp[7] for inp in input]
  lengths_user = [len(s[5]) for s in input]

  # user behavior -> item seq
  if maxlen is not None:
    new_seqs_mid = []
    new_seqs_mid_t = []
    new_seqs_cat = []
    new_lengths_x = []
    for l_x, inp in zip(lengths_mid, input):
      if l_x > maxlen:
        new_seqs_mid.append(inp[3][l_x - maxlen:])
        new_seqs_mid_t.append(inp[7][l_x - maxlen:])
        new_seqs_cat.append(inp[4][l_x - maxlen:])
        new_lengths_x.append(maxlen)
      else:
        new_seqs_mid.append(inp[3])
        new_seqs_mid_t.append(inp[7])
        new_seqs_cat.append(inp[4])
        new_lengths_x.append(l_x)
    lengths_mid = new_lengths_x
    seqs_mid = new_seqs_mid
    seqs_mid_t = new_seqs_mid_t
    seqs_cat = new_seqs_cat
    if len(lengths_mid) < 1:
      return None, None, None, None

  # dynamic item -> user seq
  if seq_user_maxlen is not None:
    new_seqs_users = []
    new_seqs_users_t = []
    new_lengths_user = []
    for l_u, inp in zip(lengths_user, input):
      if l_u > seq_user_maxlen:
        new_seqs_users.append(inp[5][l_u - seq_user_maxlen:l_u])
        new_seqs_users_t.append(inp[6][l_u - seq_user_maxlen:l_u])
        new_lengths_user.append(seq_user_maxlen)
      else:
        new_seqs_users.append(inp[5])
        new_seqs_users_t.append(inp[6])
        # seq_length user for 'default user' the id of default user is 0
        if l_u == 1 and inp[5][0] == 0:
          new_lengths_user.append(1)
        else:
          new_lengths_user.append(l_u)

    lengths_user = new_lengths_user
    seqs_user = new_seqs_users
    seqs_user_t = new_seqs_users_t

  n_samples = len(seqs_mid)
  # maxlen_item = numpy.max(lengths_mid)
  maxlen_item = maxlen
  # maxlen_u = numpy.max(lengths_user)
  maxlen_u = seq_user_maxlen

  mid_his = numpy.zeros((n_samples, maxlen_item)).astype('int32')
  cat_his = numpy.zeros((n_samples, maxlen_item)).astype('int32')
  user_his = numpy.zeros((n_samples, maxlen_u)).astype('int32')
  user_his_t = numpy.ones((n_samples, maxlen_u)).astype('int32') * -1
  mid_his_t = numpy.ones((n_samples, maxlen_item)).astype('int32') * -1

  mid_mask = numpy.zeros((n_samples, maxlen_item)).astype('float32')
  user_mask = numpy.zeros((n_samples, maxlen_u)).astype('float32')

  for idx, [s_x, s_y, u, u_t, m_t] in enumerate(zip(seqs_mid, seqs_cat, seqs_user, seqs_user_t, seqs_mid_t)):
    mid_mask[idx, :lengths_mid[idx]] = 1.
    user_mask[idx, :lengths_user[idx]] = 1.
    mid_his[idx, :lengths_mid[idx]] = s_x
    cat_his[idx, :lengths_mid[idx]] = s_y
    user_his[idx, :lengths_user[idx]] = u
    user_his_t[idx, :lengths_user[idx]] = u_t
    mid_his_t[idx, :lengths_mid[idx]] = m_t

  uids = numpy.array([inp[0] for inp in input])
  mids = numpy.array([inp[1] for inp in input])
  cats = numpy.array([inp[2] for inp in input])
  return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(
    lengths_mid), user_his, numpy.array(lengths_user), user_mask, user_his_t, mid_his_t

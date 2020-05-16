import os
import random
import sys

import numpy
from sklearn import metrics

import data_utils
from data_iterator import DataIterator
from prepare_data.path import datafilename

# EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
BATCH_SIZE = 128
MAXLEN = 100
SEQ_USER_MAXLEN = 50

best_auc = 0.0
best_auc_print = 0.0
best_loss_print = 0.0
best_accuracy_print = 0.0
best_f1_print = 0.0
best_itr_print = 0
best_iter_print = 0

from model import *


def print_metric(mode, epoch, ite, test_loss, test_accuracy, test_auc, test_f1):
  print(
      'mode: %s --- epoch: %d --- iter: %d ----> loss: %.4f ---- accuracy: %.4f ---- auc: %.4f ---- f1: %.4f' % (
    mode, epoch, ite, test_loss, test_accuracy, test_auc, test_f1))


def eval(sess, test_data, model, model_path, eval_writer):
  y_true = []
  y_pred = []
  for src, tgt in test_data:
    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, user_his, ul, user_mask, user_his_t, mid_his_t = data_utils.prepare_data(
      src, tgt, MAXLEN, SEQ_USER_MAXLEN)
    prob, loss, acc, summary = model.calculate(sess,
                                               [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, user_his, ul,
                                                user_mask, user_his_t, mid_his_t])
    prob_1 = prob[:, 0].tolist()
    target_1 = target[:, 0].tolist()
    for p, t in zip(prob_1, target_1):
      y_true.append(t)
      y_pred.append(p)
  test_auc = metrics.roc_auc_score(y_true, y_pred)
  test_f1 = metrics.f1_score(numpy.round(y_true), numpy.round(y_pred))
  test_loss = metrics.log_loss(y_true, y_pred)
  test_acc = metrics.accuracy_score(numpy.round(y_true), numpy.round(y_pred))

  global best_auc
  if best_auc < test_auc:
    best_auc = test_auc
    model.save(sess, model_path)

  eval_writer.add_summary(
    summary=tf.Summary(
      value=[tf.Summary.Value(tag='Eval AUC', simple_value=test_auc),
             tf.Summary.Value(tag='Eval Loss', simple_value=test_loss),
             tf.Summary.Value(tag='Eval ACC', simple_value=test_acc),
             tf.Summary.Value(tag='Eval F1', simple_value=test_f1)
             ]),
    global_step=model.global_step.eval())
  return test_auc, test_loss, test_acc, test_f1


def print_best_metric(itr, iter, test_loss, test_accuracy, test_auc, test_f1):
  global best_auc_print
  global best_loss_print
  global best_accuracy_print
  global best_f1_print
  global best_itr_print
  global best_iter_print
  if best_auc_print < test_auc:
    best_auc_print = test_auc
    best_loss_print = test_loss
    best_accuracy_print = test_accuracy
    best_f1_print = test_f1
    best_itr_print = itr
    best_iter_print = iter
  print('*****************************************')
  print('current best metrics')
  print_metric('test', best_itr_print, best_iter_print, best_loss_print, best_accuracy_print, best_auc_print,
               best_f1_print)
  print('*****************************************')


def train(
    batch_size=BATCH_SIZE,
    ubh_maxlen=MAXLEN,
    ibh_maxlen=SEQ_USER_MAXLEN,
    test_iter=100,
    save_iter=10000,
    model_type='DNN',
    seq_user_t=50,
    seed=2,
    learning_rate=0.001,
    epoch=2,
    dataset='Amazon_Clothing',
    emb=32,
    hidden_units='256,128,1'
):
  train_file = datafilename(dataset, "local_train_by_time")
  test_file = datafilename(dataset, "local_test_by_time")
  test_file1 = datafilename(dataset, "local_test_u1")
  test_file2 = datafilename(dataset, "local_test_u2")
  test_file3 = datafilename(dataset, "local_test_u3")
  uid_voc = datafilename(dataset, "uid_voc.pkl")
  mid_voc = datafilename(dataset, "mid_voc.pkl")
  cat_voc = datafilename(dataset, "cat_voc.pkl")
  model_path = "dnn_save_path/{}_seed{}/".format(model_type, str(seed))
  best_model_path = "dnn_best_path/{}_seed_{}/".format(model_type, str(seed))
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if not os.path.exists(best_model_path):
    os.makedirs(best_model_path)
  train_writer = tf.summary.FileWriter(model_path + '/train')
  eval_writer = tf.summary.FileWriter(model_path + '/eval')

  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, ubh_maxlen)
    test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size * 100, ubh_maxlen)
    n_uid, n_mid, n_cat = train_data.get_n()
    print "uid count : %d" % n_uid
    print "mid count : %d" % n_mid
    print "cat count : %d" % n_cat
    EMBEDDING_DIM = emb
    HIDDEN_UNITS = hidden_units.split(',')
    if model_type == 'DNN':
      model = DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    elif model_type == 'PNN':
      model = PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM)
    elif model_type == 'SVDPP':
      model = SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum=1, item_fnum=2)
    elif model_type == 'GRU4REC':
      model = GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'DIN':
      model = DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'ATRANK':
      model = ATRANK(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'CASER':
      model = CASER(n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum=1, item_fnum=2)
    elif model_type == 'DIEN':
      model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'UBGRUA':
      model = UBGRUA(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'TopoLSTM':
      model = TopoLSTM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'DIB':
      model = DIB(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'IBGRUA':
      model = IBGRUA(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'TIEN_sumagg':
      model = TIEN_sumagg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'TIEN_timeatt':
      model = TIEN_timeatt(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'TIEN_robust':
      model = TIEN_robust(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'TIEN_timeaware':
      model = TIEN_timeaware(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t)
    elif model_type == 'TIEN':
      model = TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, SEQ_USER_T=seq_user_t, HIDDEN_UNITS=HIDDEN_UNITS)
    # incorpration
    elif model_type == 'GRU4REC_TIEN':
      model = GRU4REC_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'ATRANK_TIEN':
      model = ATRANK_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    elif model_type == 'CASER_TIEN':
      model = CASER_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, user_fnum=1, item_fnum=2)
    elif model_type == 'DIEN_TIEN':
      model = DIEN_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE)
    else:
      print ("Invalid model_type : %s", model_type)
      return
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sys.stdout.flush()

    test_auc, test_loss, test_accuracy, test_f1 = eval(sess, test_data, model, best_model_path, eval_writer)
    print_metric('test', 0, 0, test_loss, test_accuracy, test_auc, test_f1)
    sys.stdout.flush()

    iter = 0
    iter_epoch = 0
    lr = learning_rate
    EPOCH = epoch
    for itr in range(EPOCH):
      y_true = []
      y_pred = []

      for src, tgt in train_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, ubh_len, user_his, ibh_len, user_mask, user_his_t, mid_his_t = data_utils.prepare_data(
          src, tgt, ubh_maxlen, ibh_maxlen)
        probs, loss, acc, summary = model.train(sess,
                                                [uids, mids, cats, mid_his, cat_his, mid_mask, target, ubh_len, lr,
                                                 user_his,
                                                 ibh_len,
                                                 user_mask, user_his_t, mid_his_t])
        prob_1 = probs[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
          y_true.append(t)
          y_pred.append(p)

        train_writer.add_summary(summary, global_step=model.global_step.eval())
        iter += 1
        iter_epoch += 1

        sys.stdout.flush()
        if (iter % test_iter) == 0 or ((itr == EPOCH - 1) and iter % (test_iter / 1) == 0):
          train_auc = metrics.roc_auc_score(y_true, y_pred)
          train_f1 = metrics.f1_score(numpy.round(y_true), numpy.round(y_pred))
          train_loss = metrics.log_loss(y_true, y_pred)
          train_acc = metrics.accuracy_score(numpy.round(y_true), numpy.round(y_pred))
          print_metric('train', itr, iter, train_loss, train_acc, train_auc, train_f1)

          test_auc, test_loss, test_accuracy, test_f1 = eval(sess, test_data, model, best_model_path, eval_writer)
          print_metric('test', itr, iter, test_loss, test_accuracy, test_auc, test_f1)
          print_best_metric(itr, iter, test_loss, test_accuracy, test_auc, test_f1)

          sys.stdout.flush()
          y_true = []
          y_pred = []
        if (iter % save_iter) == 0:
          print('save model iter: %d' % (iter))
          model.save(sess, model_path)

        # if itr == EPOCH - 1:
        #   if iter_epoch >= test_iter * 10:
        #     break

      print('*****************************************')

      test_auc, test_loss, test_accuracy, test_f1 = eval(sess, test_data, model, best_model_path, eval_writer)
      print_metric('test', itr, iter, test_loss, test_accuracy, test_auc, test_f1)
      print_best_metric(itr, iter, test_loss, test_accuracy, test_auc, test_f1)

      sys.stdout.flush()
      print('*****************************************')
      lr *= 0.5

      iter_epoch = 0


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="progrom description")
  parser.add_argument('-m', '--model', default='DNN')
  parser.add_argument('-s', '--seed', type=int, default=1234)
  parser.add_argument('-i', '--iblen', type=int, default=50, choices=[5, 10, 20, 30, 40, 50])
  parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, choices=[0.01, 0.005])
  parser.add_argument('-emb', '--embedding', type=float, default=32, choices=[8, 16, 32, 64, 128])
  parser.add_argument('-e', '--epoch', type=int, default=2, choices=[1, 2, 3, 4, 5, 10])
  parser.add_argument('-hu', '--hidden_units', default='256,128,1')
  parser.add_argument('-d', '--dataset', default="Amazon_Clothing_Shoes_and_Jewelry",
                      choices=["Amazon_Clothing_Shoes_and_Jewelry"])
  parser.add_argument('-t', '--test_iter', type=int, default=100)

  args = parser.parse_args()

  print 'model_name : %s' % args.model
  print 'seed : %d' % args.seed
  print 'iblen : %d' % args.iblen
  print 'learning_rate : %f' % args.learning_rate
  print 'epoch : %d' % args.epoch
  print 'dataset : %s' % args.dataset
  print 'test_iter : %d' % args.test_iter
  print 'emb : %d' % args.embedding
  print 'hidden_units : %s' % args.hidden_units

  SEED = args.seed
  tf.set_random_seed(SEED)
  numpy.random.seed(SEED)
  random.seed(SEED)

  train(model_type=args.model, seed=args.seed, seq_user_t=args.iblen, learning_rate=args.learning_rate,
        epoch=args.epoch,
        dataset=args.dataset,
        test_iter=args.test_iter,
        emb=args.embedding,
        hidden_units=args.hidden_units)

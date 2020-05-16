import random
import subprocess

import numpy as np

from path import datafilename

category = 'Amazon_Clothing_Shoes_and_Jewelry'

np.random.seed(1234)
random.seed(1234)


def file_len(fname):
  p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0:
    raise IOError(err)
  return int(result.strip().split()[0])


def get_cut_timestamp(train_percent=0.85):
  time_list = []

  fi = open(datafilename(category, "local_all_sample_by_time"), "r")
  path = datafilename(category, "local_all_sample_by_time")
  samples_count = file_len(path)
  train_size = int(samples_count * train_percent)
  for line in fi:
    line = line.strip()
    time = float(line.split("\t")[-1])
    time_list.append(time)
  index = np.argsort(time_list, axis=-1)
  cut_time_index = index[train_size]
  return time_list[cut_time_index]


def split_test_by_time(cut_time):
  fi = open(datafilename(category, "local_all_sample_by_time"), "r")
  ftrain = open(datafilename(category, "local_train_by_time"), "w")
  ftest = open(datafilename(category, "local_test_by_time"), "w")

  for line in fi:
    line = line.strip()
    time = float(line.split("\t")[-1])

    if time <= cut_time:
      print>> ftrain, line[:-2]
    else:
      print>> ftest, line[:-2]


def split_test_by_seqlen():
  fi = open(datafilename(category, "local_test_by_time"), "r")
  ftest_u1 = open(datafilename(category, "local_test_u1"), "w")
  ftest_u2 = open(datafilename(category, "local_test_u2"), "w")
  ftest_u3 = open(datafilename(category, "local_test_u3"), "w")

  for line in fi:
    line = line.strip()
    item_seq = line.split("\t")[4]
    sl = len(item_seq.split(""))
    if sl < 5:
      print>> ftest_u1, line
    elif sl < 15:
      print>> ftest_u2, line
    else:
      print>> ftest_u3, line


def get_all_samples():
  fin = open(datafilename(category, "jointed-new-by-time"), "r")
  fall = open(datafilename(category, "local_all_sample_by_time"), "w")
  gap = np.array([1.1, 1.4, 1.7, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

  last_user = "0"
  line_idx = 0
  for line in fin:
    items = line.strip().split("\t")
    clk = int(items[0])
    user = items[1]
    movie_id = items[2]
    dt = items[4]
    cat1 = items[5]
    user_list = items[6]
    user_t_list = items[7]

    if user != last_user:
      movie_id_list = []
      cate1_list = []
      movie_id_t_list = []
    else:
      history_clk_num = len(movie_id_list)
      cat_str = ""
      mid_str = ""
      for c1 in cate1_list:
        cat_str += c1 + ""
      for mid in movie_id_list:
        mid_str += mid + ""
      dt_gap = []
      for t in movie_id_t_list:
        temp = float(dt) / 3600.0 / 24.0 - float(t) / 3600.0 / 24.0 + 1.
        dt_gap.append(str(np.sum(temp >= gap)))
      dt_gap_str = "".join(dt_gap)
      if len(cat_str) > 0: cat_str = cat_str[:-1]
      if len(mid_str) > 0: mid_str = mid_str[:-1]

      if history_clk_num >= 1:  # 8 is the average length of user behavior
        print >> fall, items[0] + "\t" + user + "\t" + movie_id + "\t" + cat1 + "\t" + mid_str + "\t" + cat_str + \
                       "\t" + user_list + "\t" + user_t_list + '\t' + dt_gap_str + "\t" + dt
    last_user = user
    if clk:
      movie_id_list.append(movie_id)
      cate1_list.append(cat1)
      movie_id_t_list.append(dt)
    line_idx += 1


get_all_samples()
cut_time = get_cut_timestamp(train_percent=0.85)
split_test_by_time(cut_time)

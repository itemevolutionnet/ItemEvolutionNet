import cPickle
import random

import numpy as np

category = 'Amazon_Clothing_Shoes_and_Jewelry'

from path import datafilename

np.random.seed(1234)
random.seed(1234)

f_train = open(datafilename(category, "local_train_by_time"), "r").readlines()
f_test = open(datafilename(category, "local_test_by_time"), "r").readlines()

f_all = f_train + f_test

uid_dict = {}
mid_dict = {}
cat_dict = {}

iddd = 0
for line in f_all:
  arr = line.strip("\n").split("\t")
  clk = arr[0]
  uid = arr[1]
  mid = arr[2]
  cat = arr[3]
  mid_list = arr[4]
  cat_list = arr[5]
  if uid not in uid_dict:
    uid_dict[uid] = 0
  uid_dict[uid] += 1
  if mid not in mid_dict:
    mid_dict[mid] = 0
  mid_dict[mid] += 1
  if cat not in cat_dict:
    cat_dict[cat] = 0
  cat_dict[cat] += 1
  if len(mid_list) == 0:
    continue
  for m in mid_list.split(""):
    if m not in mid_dict:
      mid_dict[m] = 0
    mid_dict[m] += 1
  # print iddd
  iddd += 1
  for c in cat_list.split(""):
    if c not in cat_dict:
      cat_dict[c] = 0
    cat_dict[c] += 1

sorted_uid_dict = sorted(uid_dict.iteritems(), key=lambda x: x[1], reverse=True)
sorted_mid_dict = sorted(mid_dict.iteritems(), key=lambda x: x[1], reverse=True)
sorted_cat_dict = sorted(cat_dict.iteritems(), key=lambda x: x[1], reverse=True)

uid_voc = {}
uid_voc["default_user"] = 0
index = 1
for key, value in sorted_uid_dict:
  uid_voc[key] = index
  index += 1

mid_voc = {}
mid_voc["default_mid"] = 0
index = 1
for key, value in sorted_mid_dict:
  mid_voc[key] = index
  index += 1

cat_voc = {}
cat_voc["default_cat"] = 0
index = 1
for key, value in sorted_cat_dict:
  cat_voc[key] = index
  index += 1

cPickle.dump(uid_voc, open(datafilename(category, "uid_voc.pkl"), "w"))
cPickle.dump(mid_voc, open(datafilename(category, "mid_voc.pkl"), "w"))
cPickle.dump(cat_voc, open(datafilename(category, "cat_voc.pkl"), "w"))

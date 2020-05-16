import random

import numpy as np

from path import datafilename

gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

np.random.seed(1234)
random.seed(1234)


def process_meta(file):
  fi = open(datafilename(category, file), "r")
  fo = open(datafilename(category, "item-info"), "w")
  for line in fi:
    obj = eval(line)
    cat = obj["categories"][0][-1]
    print>> fo, obj["asin"] + "\t" + cat


def process_reviews(file):
  fi = open(datafilename(category, file), "r")
  user_map = {}
  fo = open(datafilename(category, "reviews-info"), "w")
  for line in fi:
    obj = eval(line)
    userID = obj["reviewerID"]
    itemID = obj["asin"]
    rating = obj["overall"]
    time = obj["unixReviewTime"]
    print>> fo, userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time)


def manual_join():
  f_rev = open(datafilename(category, "reviews-info"), "r")
  user_map = {}  ## User clicked on the list of items
  item_list = []  # all items list
  useridToClickItem = {}  # The user dict who clicked on the item
  for line in f_rev:
    line = line.strip()
    items = line.split("\t")
    # loctime = time.localtime(float(items[-1]))
    # items[-1] = time.strftime('%Y-%m-%d', loctime)
    if items[0] not in user_map:
      user_map[items[0]] = []
    user_map[items[0]].append(("\t".join(items), float(items[-1])))
    item_list.append(items[1])

  # The user dict who clicked on the item
  f_rev = open(datafilename(category, "reviews-info"), "r")
  for line in f_rev:
    data = line.split("\t")
    if data[1] not in useridToClickItem:
      useridToClickItem[data[1]] = []
    useridToClickItem[data[1]].append((data[0], float(data[-1])))

  f_meta = open(datafilename(category, "item-info"), "r")
  meta_map = {}  # itemID map cate
  for line in f_meta:
    arr = line.strip().split("\t")
    if arr[0] not in meta_map:
      meta_map[arr[0]] = arr[1]
      arr = line.strip().split("\t")
  fo = open(datafilename(category, "jointed-new-by-time"), "w")
  for key in user_map:
    sorted_user_bh = sorted(user_map[key], key=lambda x: x[1])
    for line, t in sorted_user_bh:
      items = line.split("\t")
      asin = items[1]
      cur_t = float(items[3]) // 3600 // 24
      j = 0
      target_user_pos_in_seq = 0
      while True:
        asin_neg_index = random.randint(0, len(item_list) - 1)
        asin_neg = item_list[asin_neg_index]
        if asin_neg == asin:
          continue
        items[1] = asin_neg

        if len(useridToClickItem[asin_neg]) == 0:
          user_str = "default_user"
          user_t_str = "-1"
        else:
          user_str = ""
          user_t_str = ""
          sorted_user_in_item_seq = sorted(useridToClickItem[asin_neg], key=lambda x: x[1])
          for i, (u, t) in enumerate(sorted_user_in_item_seq):
            if int(t) > int(items[-1]):
              target_user_pos_in_seq = i + 1
              break
            if u == items[0]:
              continue
            user_str += u + ""
            user_t = float(cur_t) - t // 3600 // 24 + 1.
            user_t_str += str(np.sum(user_t >= gap)) + ""

        if len(user_str) > 0:
          user_str = user_str[:-1]
          user_t_str = user_t_str[:-1]
        if len(user_str) == 0:
          user_str = "default_user"
          user_t_str = "-1"
        if asin_neg in meta_map:
          print>> fo, "0" + "\t" + "\t".join(items) + "\t" + meta_map[
            asin_neg] + "\t" + user_str + "\t" + user_t_str + "\t" + items[3]
        else:
          print>> fo, "0" + "\t" + "\t".join(
            items) + "\t" + "default_cat" + "\t" + user_str + "\t" + user_t_str + "\t" + items[3]

        j += 1
        if j == 1:  # negative sampling frequency
          break

      target_user_pos_in_seq = 0
      # useridToClickItem[asin][0].remove(items[0])
      if len(useridToClickItem[asin]) == 0:
        user_str = "default_user"
        user_t_str = "-1"
      else:
        user_str = ""
        user_t_str = ""
        sorted_user_in_item_seq = sorted(useridToClickItem[asin], key=lambda x: x[1])
        for i, (u, t) in enumerate(sorted_user_in_item_seq):
          if int(t) > int(items[-1]):
            target_user_pos_in_seq = i
            break
          if u == items[0]:
            continue
          user_str += u + ""
          user_t = float(cur_t) - t // 3600 // 24 + 1.
          user_t_str += str(np.sum(user_t >= gap)) + ""
      if len(user_str) > 0:
        user_str = user_str[:-1]
        user_t_str = user_t_str[:-1]
      if len(user_str) == 0:
        user_str = "default_user"
        user_t_str = "-1"
      if asin in meta_map:
        print>> fo, "1" + "\t" + line + "\t" + meta_map[asin] + "\t" + user_str + "\t" + user_t_str + "\t" + items[3]
      else:
        print>> fo, "1" + "\t" + line + "\t" + "default_cat" + "\t" + user_str + "\t" + user_t_str + "\t" + items[3]


category = 'Amazon_Clothing_Shoes_and_Jewelry'
process_meta('meta_Clothing_Shoes_and_Jewelry.json')
process_reviews('reviews_Clothing_Shoes_and_Jewelry_5.json')
manual_join()

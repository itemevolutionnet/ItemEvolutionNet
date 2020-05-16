import cPickle as pkl
import gzip
import json


def unicode_to_utf8(d):
  return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
  try:
    with open(filename, 'rb') as f:
      return unicode_to_utf8(json.load(f))
  except:
    with open(filename, 'rb') as f:
      return unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
  if filename.endswith('.gz'):
    return gzip.open(filename, mode)
  return open(filename, mode)


class DataIterator:
  def __init__(self, source,
               uid_voc,
               mid_voc,
               cat_voc,
               batch_size=128,
               skip_empty=False,
               max_batch_size=20000,
               minlen=None):

    self.source = fopen(source, 'r')
    self.source_dicts = []
    for source_dict in [uid_voc, mid_voc, cat_voc]:
      self.source_dicts.append(load_dict(source_dict))

    self.batch_size = batch_size
    self.minlen = minlen
    self.skip_empty = skip_empty

    self.n_uid = len(self.source_dicts[0])
    self.n_mid = len(self.source_dicts[1])
    self.n_cat = len(self.source_dicts[2])

    self.source_buffer = []
    self.k = batch_size * max_batch_size

    self.end_of_data = False

  def get_n(self):
    return self.n_uid, self.n_mid, self.n_cat

  def __iter__(self):
    return self

  def reset(self):
    self.source.seek(0)

  def next(self):
    if self.end_of_data:
      self.end_of_data = False
      self.reset()
      raise StopIteration

    source = []
    target = []

    if len(self.source_buffer) == 0:
      for k_ in xrange(self.k):
        ss = self.source.readline()
        if ss == "":
          break
        self.source_buffer.append(ss.strip("\n").split("\t"))

    if len(self.source_buffer) == 0:
      self.end_of_data = False
      self.reset()
      raise StopIteration

    try:

      # actual work here
      while True:

        # read from source file and map to word index
        try:
          ss = self.source_buffer.pop()
        except IndexError:
          break

        uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
        mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
        cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
        tmp = []
        for fea in ss[4].split(""):
          m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
          tmp.append(m)
        mid_list = tmp

        tmp1 = []
        for fea in ss[5].split(""):
          c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
          tmp1.append(c)
        cat_list = tmp1

        tmp2 = []
        arr = ss[6].split("")
        for fea in arr:
          c = self.source_dicts[0][fea] if fea in self.source_dicts[0] else 0
          tmp2.append(c)
        user_list = tmp2

        # time of user seq
        arr = ss[7].split("")
        time_list = arr

        # time of item seq
        arr = ss[8].split("")
        item_time_list = arr

        # read from source file and map to word index
        if self.minlen != None:
          if len(mid_list) <= self.minlen:
            continue
        if self.skip_empty and (not mid_list):
          continue

        source.append([uid, mid, cat, mid_list, cat_list, user_list, time_list, item_time_list])
        target.append([float(ss[0])])

        if len(source) >= self.batch_size or len(target) >= self.batch_size:
          break
    except IOError:
      self.end_of_data = True

    # all sentence pairs in maxibatch filtered out because of length
    if len(source) == 0 or len(target) == 0:
      source, target = self.next()

    return source, target

import os

process_data_path = os.path.split(os.path.realpath(__file__))[0]


def datafilename(datapath, file_name):
  return process_data_path + "/../../../dataset/" + datapath + "/" + file_name

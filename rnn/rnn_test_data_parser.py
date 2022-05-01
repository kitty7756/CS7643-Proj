#!/usr/bin/python

import sys
import ujson as json
import pandas as pd
import os
import glob
import random
import time
from rnn_util import read_rnn_data
from rnn_dataset import RNNDataset

usage = "python rnn_test_data_parser.py"

def convert_data():
  transformed_dataset = RNNDataset('../data/parsed_data/test_parsed_data.pkl')
  item_arr = []
  for i in range(2):
    item_arr.append(transformed_dataset.__getitem__(i))
  df = pd.DataFrame(item_arr, columns = ['drawing', 'class'])
  df.to_pickle(os.path.join('../data/test_parser_data', "sample_rnn_data.pkl"))

convert_data()

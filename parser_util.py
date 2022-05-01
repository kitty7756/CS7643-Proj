#!/usr/bin/python

import sys
import ujson as json
import pandas as pd
import os
import glob
import random
import time
from rnn.rnn_parser import parse_rnn_dataframe

startTime = time.time()
usage = "python parser_util.py RNN rnn_parsed_data"

raw_data_path = './data/test_parser_data'
# Uncomment for raw_data
raw_data_path = './data/raw_data'
parsed_data_path = './data/parsed_data'
filenames = glob.glob(os.path.join(raw_data_path, '*.ndjson'))
filenames = sorted(filenames)

# Simplified Drawing files (.ndjson)
# We've simplified the vectors, removed the timing information, and positioned and scaled the data into a 256x256 region. The data is exported in ndjson format with the same metadata as the raw format. The simplification process was:

# Align the drawing to the top-left corner, to have minimum values of 0.
# Uniformly scale the drawing, to have a maximum value of 255.
# Resample all strokes with a 1 pixel spacing.
# Simplify all strokes using the Ramer–Douglas–Peucker algorithm with an epsilon value of 2.0.
# There is an example in examples/nodejs/simplified-parser.js showing how to read ndjson files in NodeJS.
# Additionally, the examples/nodejs/ndjson.md document details a set of command-line tools that can help explore subsets of these quite large files.

def convert_data(data_type, save_as_filename):
  all_data = []
  for filename in filenames:
    df = pd.read_json(filename, lines=True)
    if data_type == "RNN":
      all_data += parse_rnn_dataframe(df)
    else:
      # call CNN conversion method
      df = df

  # Randomize data
  random.shuffle(all_data)

  # Write to file in to_pickle
  df = pd.DataFrame(all_data, columns = ['drawing', 'class'])
  df.to_pickle(os.path.join(parsed_data_path, save_as_filename + ".pkl"))

args = sys.argv
if len(args) != 3:
  print('Invalid usage. Usage: ', usage)
  sys.exit()

convert_data(args[1], args[2])

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

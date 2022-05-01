import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rnn_dataset import RNNDataset

# Returns a DataLoader
# It already shuffles data as per sampler option
def read_rnn_data(batch_size=4, num_workers=0):
	transformed_dataset = RNNDataset('../data/parsed_data/rnn_parsed_data.pkl')
	def collate_fn(batch):
    	return tuple(zip(*batch))
	return DataLoader(transformed_dataset, batch_size, num_workers, collate_fn=collate_fn)
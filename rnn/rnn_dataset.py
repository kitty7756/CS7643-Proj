from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

class RNNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, parsed_data, transform=None):
        """
        Args:
            parsed_data (string): Path to the parsed_data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_pkg = pd.read_pickle(parsed_data)
        self.transform = transform

    def __len__(self):
        return len(self.file_pkg)

    def __getitem__(self, idx):
        return self.file_pkg.iloc[idx].to_dict()

# Example usage: 
# transformed_dataset = RNNDataset(csv_file='data/parsed_data/rnn_parsed_data.csv')

# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched)
# 
# more tutorials here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
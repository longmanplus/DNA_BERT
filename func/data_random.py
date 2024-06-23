import torch
import lightning as L

# --------------------------------
#  Create data for testing
# --------------------------------
class GenerateDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        n_ids = config.n_ids
        n_classes = config.n_classes
        seq_len = config.seq_len
        sample_size = config.sample_size
        self.sample_size = sample_size
        self.ids = []
        self.input_values = []
        self.target_values = []
        for _ in range(sample_size):
            ids = torch.arange(n_ids).reshape(1, n_ids)
            values = torch.randint(low=1, high=n_classes, size=(1, seq_len))
            # make 2nd half of sequence equal to 1st half
            half_len = seq_len // 2
            values[:, half_len:] = values[:, :half_len].flip(1)
            # masking 15% of values as zeros
            mask = torch.rand(size=(1, seq_len)) < config.mask_frac
            masked_values = torch.where(mask, torch.zeros_like(values), values)
            self.ids.append(ids)
            self.input_values.append(masked_values)
            self.target_values.append(values)
        self.ids = torch.cat(self.ids)
        self.input_values = torch.cat(self.input_values)
        self.target_values = torch.cat(self.target_values)

    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        ids = self.ids[idx]
        input_values = self.input_values[idx]
        target_values = self.target_values[idx]
        return ids, input_values, target_values

from torch.utils.data import random_split, DataLoader

class GeneratedDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        
    def setup(self, stage: str):
        ds_full = GenerateDataset(self.config)
        self.ds_train, self.ds_val = random_split(
            ds_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=15, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=15)


import torch, torch.nn as nn, torch.nn.functional as F
import math
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('func')
from func.encoder_pytorch_modified import get_encoder
from config import config

# --------------------------------
#  Define BERT model
# --------------------------------
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class embedding_int(nn.Module):
    def __init__(self, max_int, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.embed = nn.Embedding(max_int, d_embed)
    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_embed)
        embeded = self.embed(x) * math.sqrt(self.d_embed) 
        return embeded

class pl_model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        d_embed_id = config.d_model
        d_embed_class = config.d_model
        d_model = config.d_model
        self.config = config
        self.id_embedding = embedding_int(config.n_ids, d_embed_id) 
        self.class_embedding = embedding_int(config.n_classes, d_embed_class)
        self.correct_dim = nn.Sequential(
            nn.Linear(d_embed_id+d_embed_class, d_model),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.encoder = get_encoder(config)
        # self.decoder = nn.Linear(d_model, 1)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(d_model, config.n_classes),
        )
        self.validation_outputs = []
        
    def forward(self, id, x):
        # x.shape: (batch, seq_len)        
        batch, seq_len = x.shape
        id = self.id_embedding(id)
        x = self.class_embedding(x)
        # x.shape: (batch, seq_len, d_embed)        
        x = torch.cat([id, x], dim=2)
        x = self.correct_dim(x)
        # x = x + id
        x = self.encoder(x)
        y_hat = self.decoder(x)
        return y_hat
    
    def forward_w_attn(self, id, x):
        batch, seq_len = x.shape
        id = self.id_embedding(id)
        x = self.class_embedding(x)
        # x.shape: (batch, seq_len, d_embed)        
        x = torch.cat([id, x], dim=2)
        x = self.correct_dim(x)
        # x = x + id
        x, attn_maps = self.encoder.forward_w_attn(x)
        y_hat = self.decoder(x)
        return y_hat, attn_maps
    
    def _calculate_loss_acc(self, batch):
        id, x, y = batch
        y_hat = self(id, x)
        y = y.reshape(-1)
        y_hat = y_hat.reshape(-1, config.n_classes)
        # only select masked position to calculate loss and acc
        mask = x == 0
        mask = mask.reshape(-1)
        y = y[mask]
        # add 1 dim to make it 2 dim
        # mask = mask.unsqueeze(-1)
        # repeat over 2nd dim
        # mask = mask.repeat(1, config.n_classes)
        y_hat = y_hat[mask,:]
        loss = F.cross_entropy(y_hat, y)
        acc = y_hat.argmax(dim=1).eq(y).sum().item() / y.shape[0]
        return(loss, acc)
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss_acc(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=False)
        self.log("train/acc", acc, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        id, x, y = batch
        y_hat = self(id, x)
        y = y.reshape(-1) 
        y_hat = y_hat.reshape(-1, config.n_classes)
        # only select masked position to calculate loss and acc
        mask = x == 0
        mask = mask.reshape(-1)
        y = y[mask]
        y_hat = y_hat[mask,:]

        self.validation_outputs.append((y_hat, y))
        loss = F.cross_entropy(y_hat, y)
        acc = y_hat.argmax(dim=1).eq(y).sum().item() / y.shape[0]
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_start(self):
        self.validation_outputs = []
    
    def on_validation_epoch_end(self):
        y_hat, y = zip(*self.validation_outputs)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)
        y = y.reshape(-1)
        y_hat = y_hat.reshape(-1, config.n_classes)
        acc = y_hat.argmax(dim=1).eq(y).sum().item() / y.shape[0]
        # add a little bit noise 
        y_hat = y_hat.argmax(dim=1).float()
        y = y.float()
        y_hat = y_hat + torch.randn_like(y_hat) * 0.1
        y = y + torch.randn_like(y) * 0.1
        # scatter plot y_hat vs y
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.figure()
        plt.scatter(y.detach().cpu().numpy().flatten(), y_hat.detach().cpu().numpy().flatten(), alpha=0.3, s=0.3)
        plt.xlabel("expectation")
        plt.ylabel("prediction")
        plt.title(f"y vs y_hat, epoch: {self.current_epoch}, corr: {acc:.3f}")
        # plt.show()
        plt.savefig("results/y_vs_y_hat.png")
        plt.close()
        
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, acc = self._calculate_loss_acc(batch)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.fix_lr:
            return optimizer
        else:
            # Apply lr scheduler per step
            lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.config.lr_warmup,
                                             max_iters=self.config.lr_max_iters)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

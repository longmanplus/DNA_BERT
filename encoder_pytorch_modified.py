import torch
import torch.nn as nn
import math
import torch.nn.functional as F
        
class EncoderBlock(nn.Module):
    
    def __init__(self, n_heads, d_model, dim_feedforward, dropout=0.0):
        """
        Inputs:
            d_model - Dimensionality of the input
            n_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feedforward
        self.linear_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention 
        attn_out, _ = self.self_attn(x, x, x)

        # Dropout, ADD, Normalization
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feedforward
        linear_out = self.linear_net(x)

        # Dropout, ADD, Normalization
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, n_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    
    def forward_w_attn(self, x, mask=None):
        map_list = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, x, x, average_attn_weights=False)
            # attn_map is the attention weights for each head. shape: (batchs, heads, seq_len, seq_len)
            map_list.append(attn_map)
            x = l(x, mask=mask)  # x went through the iterations of layers
        attention_maps = torch.stack(map_list)
        # attention_maps is the attention weights for each layer. shape: (layers, batchs, heads, seq_len, seq_len)
        return x, attention_maps

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, x, x, average_attn_weights=False)
            attention_maps.append(attn_map)
            x = l(x, mask=mask)  # x went through the iterations of layers
        return attention_maps

def get_encoder(args):
    return TransformerEncoder(n_layers=args.n_layers,
                              n_heads=args.n_heads,
                              d_model=args.d_model,
                              dim_feedforward=args.d_ffn,
                              dropout=args.dropout)
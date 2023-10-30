import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import (block_size, dropout, embed_size, end_token, n_heads,
                    n_layer, unknown_token)
from src.utils import encode


class Head(nn.Module):
    """
    This module performs self-attention operations on the input tensor, producing 
    an output tensor with the same time-steps but different channels. 
    
    :param head_size: The size of the head in the multi-head attention mechanism.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        """
        B,T,C = x.shape
        k = self.key(x)                                     # (B, T, head_size)
        q = self.query(x)                                   # (B, T, head_size)

        # compute attention scores
        wei = q @ k.transpose(-2,-1)                        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei /= math.sqrt(k.shape[-1])                       # (B, T, T)
        
        # avoid look-ahead
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float("-inf"))     # (B, T, T)
        wei = F.softmax(wei, dim=-1)                        # (B, T, T)
        wei = self.dropout(wei)
        
        # weighted aggregation of the values
        v = self.value(x)                                   # (B, T, head_size)
        out = wei @ v                                       # (B, T, T) @ (B, T, hs) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    This class contains multiple `Head` objects, which perform self-attention 
    operations in parallel.
    """
    def __init__(self):
        super().__init__()

        # list of parallel heads that are concatenated by the linear layer in the end
        head_size = embed_size // n_heads
        heads_list = [Head(head_size) for _ in range(n_heads)]
        
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(n_heads * head_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads_list = [h(x) for h in self.heads]
        out = torch.cat(heads_list, dim=-1)
        out = self.linear(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """
    This module passes the input tensor through a series of linear transformations 
    and non-linear activations.
    """
    def __init__(self):
        super().__init__()
        # factor of 4 is the multiplier of nodes
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), 
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    This module contains a single transformer block, which consists of multi-head 
    self-attention followed by feed-forward neural networks.
    """
    def __init__(self):
        super().__init__()

        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    This class encompasses the entire GPT model, including the token and position embeddings, 
    multiple transformer blocks, and output layer.
    """
    def __init__(self, vocab_size: int):
        super().__init__()

        # embedding tables for token and their positioning in the context
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        
        # put one block after the other sequentially (not parallel like multi-head attention)
        block_list = [Block() for _ in range(n_layer)]
        self.blocks = nn.Sequential(*block_list)
        
        # output layer after sequential blocks
        self.ln_output = nn.LayerNorm(embed_size)
        self.linear_output = nn.Linear(embed_size, vocab_size)

        # initialize weights and biases for linear layers and embeddings
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # The linear layers in self-attention do not have a biases
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                     # (B, T, C)
        pos_emb = self.pos_embedding(torch.arange(T))           # (T, C)
        x = tok_emb + pos_emb                                   # (B, T, C)
        x = self.blocks(x)                                      # (B, T, C)
        x = self.ln_output(x)                                   # (B, T, C)
        logits = self.linear_output(x)                          # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, vocab):
        
        # Initialize idx_net for while loop
        idx_next = torch.zeros(1)
        idx_end = encode([end_token], vocab)
        idx_unk = encode([unknown_token], vocab)

        # continue to sample tokens until special end token
        while idx_next[0] != idx_end:

            # idx is (B, T) array of indices in the current context
            # crop idx to the last block_size tokens for each batch (row)
            idx_cond = idx[:, -block_size:]                     # (B, T)

            # get the predictions
            logits, _ = self(idx_cond)                          # (B, T, vocab_size)
            logits = logits[:, -1, :]                           # (B, vocab_size)            
            probs = F.softmax(logits, dim=-1)                   # (B, vocab_size)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # when the sampled token is UNK, then sample again
            while idx_next[0] == idx_unk:
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        # output everything except the end token
        return idx[0][:-1]

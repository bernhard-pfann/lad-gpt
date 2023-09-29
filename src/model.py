import math
import torch
import torch.nn as nn
from torch.nn import functional as F


dropout = 0.2
n_embd = 32
block_size = 128
vocab_size = 65 # TODO: Make variable


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
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
    """ multiple heads of self-attention in parallel """

    def __init__(self, head_size):
        super().__init__()
        # best practice to depend head_size on n_embed and n_heads
        # this is to keep compute complexity the same
        head_size = n_embd // 4

        self.head1 = Head(n_embd, head_size)
        self.head2 = Head(n_embd, head_size)
        self.head3 = Head(n_embd, head_size)
        self.head4 = Head(n_embd, head_size)

        self.linear = nn.Linear(head_size * 4, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads_out = [
            self.head1(x), 
            self.head2(x), 
            self.head3(x), 
            self.head4(x)
        ]
        out = torch.cat(heads_out, dim=-1)
        out = self.linear(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # factor of 4 is magic number from paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # factor of 4 is magic number from paper
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // 4
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # embedding tables for token and their positioning in the context
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        # put one block after the other sequentially (not parallel like multi-head attention)
        self.blocks = nn.Sequential(
            Block(n_embd),
            Block(n_embd),
            Block(n_embd),
            Block(n_embd),
            Block(n_embd)
        )
        # final output layer norm
        self.ln_output = nn.LayerNorm(n_embd)
        self.linear_output = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # The linear layers in self-attention do not have a bias
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding(idx)                 # (B, T, C)
        pos_emb = self.pos_embedding(torch.arange(T))       # (T, C)
        x = tok_emb + pos_emb                               # (B, T, C)
        x = self.blocks(x)                                  # (B, T, C)
        x = self.ln_output(x)                               # (B, T, C)
        logits = self.linear_output(x)                      # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

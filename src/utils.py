import torch
from config import eval_iters, block_size, batch_size


@torch.no_grad()
def estimate_loss(model, data):
    """
    Set evaluation mode and evaluate the loss on multiple batches. 
    Return the average of collected losses.
    """
    model.eval() 
    loss_list = torch.zeros(eval_iters)
    
    for i in range(eval_iters):
        X, Y = get_batch(data)
        logits, loss = model(X, Y)
        loss_list[i] = loss.item()

    loss_avg = loss_list.mean()    
    model.train() 
    return loss_avg


def get_batch(data):
    """Generate a small batch of data of inputs x and targets y"""

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def encode(s: str, vocab: list) -> torch.tensor:
    """Encode a string into a tensor of integers, given a fixed vocabulary."""

    map = {s:i for i,s in enumerate(vocab)}
    enc = [map[c] for c in s]
    enc = torch.tensor(enc, dtype=torch.long)
    return enc


def decode(tensor: torch.tensor, vocab: list) -> str:
    """Decode a tensor of integers, back into a string."""

    map_enc = {s:i for i,s in enumerate(vocab)}
    map_dec = {i:s for s,i in map_enc.items()}
    dec = [map_dec[i.item()] for i in tensor]
    dec = "".join(dec)
    return dec


def get_prompt(vocab: str) -> torch.tensor:
    """Get user input and encode into tensor."""

    string = input() or ""
    tensor = encode(string, vocab).unsqueeze(1).T
    return tensor

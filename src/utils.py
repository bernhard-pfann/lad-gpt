import torch
from src.config import eval_iters, block_size, batch_size


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

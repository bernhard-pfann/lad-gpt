import json
import torch

from model import GPTLanguageModel
from utils import estimate_loss, get_batch, current_time
from config import learn_rate, max_iters, eval_interval


if __name__ == "__main__":

    train_data = torch.load("../assets/train.pt")
    valid_data = torch.load("../assets/valid.pt")

    with open("../assets/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    # initialize model & optimizer
    model = GPTLanguageModel(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # number of model parameters
    n_params = sum(p.numel() for p in model.parameters())
    print("Model is being trained...")
    print(n_params, "parameters to be optimized\n", )

    # learning iterations
    for i in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0 or i == max_iters - 1:
            train_loss = estimate_loss(model, train_data)
            valid_loss = estimate_loss(model, valid_data)

            time = current_time()
            print(f"{time} | step {i}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

        # sample batch of data
        x_batch, y_batch = get_batch(train_data)

        # evaluate the loss
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        torch.save(model, "../assets/model.pt")

    print("Model saved")

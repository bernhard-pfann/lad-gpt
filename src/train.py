import json

import torch

from config import eval_interval, learn_rate, max_iters
from src.model import GPTLanguageModel
from src.utils import current_time, estimate_loss, get_batch


def model_training(update: bool) -> None:
    """
    Trains or updates a GPTLanguageModel using pre-loaded data.

    This function either initializes a new model or loads an existing model based
    on the `update` parameter. It then trains the model using the AdamW optimizer
    on the training and validation data sets. Finally the trained model is saved.

    :param update: Boolean flag to indicate whether to update an existing model.
    """
    # LOAD DATA -----------------------------------------------------------------

    train_data = torch.load("assets/output/train.pt")
    valid_data = torch.load("assets/output/valid.pt")

    with open("assets/output/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    # INITIALIZE / LOAD MODEL ---------------------------------------------------

    if update:
        try:
            model = torch.load("assets/models/model.pt")
            print("Loaded existing model to continue training.")
        except FileNotFoundError:
            print("No existing model found. Initializing a new model.")
            model = GPTLanguageModel(vocab_size=len(vocab))
        
    else:
        print("Initializing a new model.")
        model = GPTLanguageModel(vocab_size=len(vocab))

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # number of model parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters to be optimized: {n_params}\n", )

    # MODEL TRAINING ------------------------------------------------------------

    for i in range(max_iters):

        # evaluate the loss on train and valid sets every 'eval_interval' steps
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

    torch.save(model, "assets/models/model.pt")
    print("Model saved")

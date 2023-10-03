import json

import torch
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from src.config import end_token
from src.utils import decode, encode, print_delayed, tokenizer


def conversation():
    """Sampling from the trained model to emulate chat conversations."""

    with open("assets/output/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    with open("assets/output/senders.txt", "r", encoding="utf-8") as f:
        all_senders = json.loads(f.read())    

    model = torch.load("assets/output/model.pt")
    completer = WordCompleter(all_senders, ignore_case=True)
    next_sender = prompt("Sender: ", completer=completer, default="")

    while next_sender != end_token:

        tokens = tokenizer(next_sender, all_senders)
        context = encode(tokens, vocab).unsqueeze(1).T
        sampled = model.generate(context, vocab)

        print_delayed(decode(sampled, vocab))
        next_sender = prompt("Sender: ", completer=completer, default="")

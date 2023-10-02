import json

import torch
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from utils import decode, encode, tokenizer

if __name__ == "__main__":

    with open("../assets/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    with open("../assets/senders.txt", "r", encoding="utf-8") as f:
        all_senders = json.loads(f.read())    

    model = torch.load("../assets/model.pt")
    completer = WordCompleter(all_senders, ignore_case=True)
    next_sender = prompt("Sender: ", completer=completer)    

    while next_sender != "<END>":
        
        if next_sender == "": 
            next_sender = " "
        
        tokens = tokenizer(next_sender, all_senders)
        context = encode(tokens, vocab).unsqueeze(1).T
        sampled = model.generate(context, vocab)

        print(decode(sampled, vocab))
        next_sender = prompt("Sender: ", completer=completer)

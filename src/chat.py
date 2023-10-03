import json

import torch
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from src.config import end_token
from src.utils import decode, encode, print_delayed, tokenizer


def conversation() -> None:
    """
    Emulates chat conversations by sampling from a pre-trained GPTLanguageModel.

    This function loads a trained GPTLanguageModel along with vocabulary and 
    the list of chat participants 'sender names'. It then enters into a loop 
    where the user specifies a sender. Given this input, the model generates a sample response.
    The conversation continues until the user inputs the end token.

    :example:

    >>> conversation()
    Sender: Alice
    Model's Response: How are you?
    Sender: end
    """
    with open("assets/output/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    with open("assets/output/senders.txt", "r", encoding="utf-8") as f:
        all_senders = json.loads(f.read())    

    model = torch.load("assets/output/model.pt")
    completer = WordCompleter(all_senders, ignore_case=True)
    
    input = prompt("sender: ", completer=completer, default="")
    output = torch.tensor([], dtype=torch.long)

    while input != end_token:

        add_tokens = tokenizer(input, all_senders)
        add_context = encode(add_tokens, vocab)
        context = torch.cat((output, add_context)).unsqueeze(1).T
        
        n0 = len(output)
        output = model.generate(context, vocab)
        n1 = len(output)

        print_delayed(decode(output[n0-n1:], vocab))
        input = prompt("sender: ", completer=completer, default="")

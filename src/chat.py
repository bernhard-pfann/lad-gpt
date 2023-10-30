import json
import random

import torch
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from config import end_token, n_chats
from src.utils import custom_tokenizer, decode, encode, print_delayed


def conversation() -> None:
    """
    Emulates chat conversations by sampling from a pre-trained GPTLanguageModel.

    This function loads a trained GPTLanguageModel along with vocabulary and 
    the list of special tokens. It then enters into a loop where the user specifies 
    a contact. Given this input, the model generates a sample response. The conversation 
    continues until the user inputs the end token.

    :example:

    >>> conversation()
    message >> Alice
    Model's Response: How are you?
    response >> end
    """
    with open("assets/output/vocab.txt", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())

    with open("assets/output/contacts.txt", "r", encoding="utf-8") as f:
        contacts = json.loads(f.read())   

    spec_tokens = contacts + [end_token]
    model = torch.load("assets/models/model.pt")
    completer = WordCompleter(spec_tokens, ignore_case=True)
    
    input = prompt("message >> ", completer=completer, default="")
    output = torch.tensor([], dtype=torch.long)
    print()

    while input != end_token:
        for _ in range(n_chats):

            add_tokens = custom_tokenizer(input, spec_tokens)
            add_context = encode(add_tokens, vocab)
            context = torch.cat((output, add_context)).unsqueeze(1).T

            n0 = len(output)
            output = model.generate(context, vocab)
            n1 = len(output)

            print_delayed(decode(output[n0-n1:], vocab))
            input = random.choice(contacts)

        input = prompt("\nresponse >> ", completer=completer, default="")
        print()
        
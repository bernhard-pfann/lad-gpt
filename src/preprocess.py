import json
import re
from collections import Counter
from typing import List, Tuple

import torch

from config import end_token
from src.utils import custom_tokenizer, encode, get_vocab


def drop_chars(txt: str, drop: List[str]) -> str:
    """Drop a list of characters from string"""

    return txt.translate(str.maketrans("", "", "".join(drop)))


def get_infrequent_chars(txt: str, min_count: int) -> List[str]:
    """Identify characters that appear less than a minimum count"""

    chars_counts = Counter(txt)
    chars_remove = [k for k,v in chars_counts.items() if v<= min_count]
    return chars_remove


def flatten_tuple(txt: List[Tuple[str, str]]) -> str:
    """Convert list of tuples into string separated by the end token"""

    return "".join([x0+":"+x1+end_token for x0, x1 in txt])


def make_train_test() -> None:
    """
    Prepare training and testing datasets from chat messages. This function performs multiple tasks:
    
    1. Reads a corpus of WhatsApp chat messages from a text file
    2. Filters out infrequent characters from the corpus
    3. Splits the text based on regular expressions
    4. Tokenizes the text and encodes the tokens into integers
    5. Splits the encoded data into training and validation sets
    6. Saves the training and validation datasets, as well as the vocab and senders, to disk
    """
    with open("data/input/chat.txt", "r") as f:
        text = f.read()

    # remove very rare characters (probably non-utf8)
    infreq_chars = get_infrequent_chars(text, min_count=400)
    infreq_chars += ["~\u202f"]
    text = drop_chars(text, infreq_chars)

    # split string into list of tuples (date, contact, message)
    pattern = r'\[(.*?)\] (.*?): (.*)'
    matches = re.findall(pattern, text)
    text = [(x1, x2.lower()) for x0, x1, x2 in matches]

    # get list of all contacts, treated as special tokens
    contacts = list(set([contact+":" for contact, msg in text]))
    spec_tokens = contacts + [end_token]

    # convert list of tuples into list of tokens (word or character level)
    text_flat = flatten_tuple(text)
    tokens = custom_tokenizer(txt=text_flat, spec_tokens=spec_tokens)

    # get vocabulary of corpus to file
    vocab = get_vocab(tokens)
    print(f"The corpus has {len(vocab)} unique tokens.")

    # encode tokens into a tensor of integers
    data = encode(tokens, vocab)

    # split up the data into train and validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # export tensors
    torch.save(train_data, "data/output/train.pt")
    torch.save(valid_data, "data/output/valid.pt")

    with open("data/output/vocab.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab))

    with open("data/output/contacts.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(contacts))

    print("SUCCESS")

import json
import re
from collections import Counter
from typing import List, Union

import torch

from utils import encode


def drop_datetime(txt: str) -> str:
    """Strip away the datetime information of each message."""
    
    txt = txt.replace("\n", "")
    date_fmt = "\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\] "
    txt_split = re.split(date_fmt, txt)[1:]

    end_token = "<END>"
    txt_split = [msg + end_token for msg in txt_split]
    txt_joined = "".join(txt_split)

    return txt_joined


def get_infrequent_chars(txt: str, min_count: int) -> List[str]:
    """Identify characters that appear less than a minimum count."""

    chars_counts = Counter(txt)
    chars_remove = [k for k,v in chars_counts.items() if v< min_count]
    return chars_remove


def drop_chars(txt: str, drop: List[str]) -> str:
    """Drop a list of characters from string."""

    return txt.translate(str.maketrans("", "", "".join(drop)))


def get_vocab(text: Union[List[str], str]) -> List[str]:
    """Returns a sorted list of all unique tokens in the corpus."""

    return sorted(list(set(text)))


def get_all_senders(text: str) -> List[str]:
    """Retrieve list of all senders."""

    text = text.split("<END>")[:-1]
    senders = [msg.split(":", 1)[0] for msg in text]
    return list(set(senders))


def tokenizer(txt: str, senders: list) -> list:
    """
    Treats all single characters as token. As an except the sender names are also
    considered single tokens each.
    """
    regex = "|".join(senders)+"|\S|\s"
    tokens = re.findall(regex, txt)
    return tokens


if __name__ == "__main__":

    # read corpus of whatsapp chat messages
    with open("../assets/chat.txt", "r") as f:
        text = f.read()

    text = drop_chars(text, "~\u202f")
    text = drop_datetime(text)

    # shrink vocabulary by eliminating rare characters
    infreq_chars = get_infrequent_chars(text, min_count=100)
    text = drop_chars(text, infreq_chars)

    # obtain list of all senders
    senders = get_all_senders(text)

    # convert text from string to list (each element is a token)
    tokens = tokenizer(text, senders)

    # get vocabulary of corpus to file
    vocab = get_vocab(tokens)

    # encode characters into a tensor of integers
    data = encode(text, vocab)

    # split up the data into train and validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # export tensors
    torch.save(train_data, "../assets/train.pt")
    torch.save(valid_data, "../assets/valid.pt")

    with open("../assets/vocab.txt", "w") as f:
        f.write(json.dumps(vocab))

    with open("../assets/senders.txt", "w") as f:
        f.write(json.dumps(senders))
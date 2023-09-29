import torch


def get_vocab(text: str) -> list:
    """Returns a sorted list of all available characters in the corpus."""

    return sorted(list(set("".join(text))))


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


if __name__ == "__main__":

    # load corpus and obtain fixed set of vocabulary from it
    text = open("assets/input.txt", "r").read()
    vocab = get_vocab(text)
    
    # write vocabulary
    with open("assets/vocab.txt", "w") as file: 
        file.write("".join(vocab))

    # encode characters into a tensor of integers
    data = encode(text, vocab)

    # split up the data into train and validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # export tensors
    torch.save(train_data, "assets/train.pt")
    torch.save(valid_data, "assets/valid.pt")
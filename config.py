# model hyperparameters
block_size = 16 # 32
embed_size = 16 # 256
dropout = 0.2
n_heads = 6
n_layer = 6
eval_iters = 200
batch_size = 32

# learning hyperparameters
learn_rate = 3e-4
max_iters = 5000
eval_interval = 500

# encoding
end_token = "<END>"
unknown_token = "<UNK>"

# tokenizer patterns
token_level = "char"
token_pattern = {"word": "|\d|\\w+|[^\\s]", "char": "|."}[token_level]

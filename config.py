# model hyperparameters
block_size = 32
embed_size = 256
dropout = 0.2
n_heads = 6
n_layer = 6
eval_iters = 200
batch_size = 32

# learning hyperparameters
learn_rate = 3e-4
max_iters = 5000
eval_interval = 500

# preprocess
min_count_chars = 800
min_count_tokens = 2

# encoding
end_token = "<END>"
unknown_token = "<UNK>"
n_chats = 5

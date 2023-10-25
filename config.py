# model hyperparameters
block_size = 32 # 128
embed_size = 128 # 128
dropout = 0.3
n_heads = 6
n_layer = 6
eval_iters = 200
batch_size = 64

# learning hyperparameters
learn_rate = 3e-4
max_iters = 1_000
eval_interval = 500

# encoding
end_token = "<END>"
# tokenizer_pattern = "|\d|\\w+|[^\\s]"
# tokenizer_pattern = "|."
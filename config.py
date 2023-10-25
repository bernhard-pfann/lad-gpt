# model hyperparameters
block_size = 16 # 128
embed_size = 32 # 128
dropout = 0.2
n_heads = 6
n_layer = 6
eval_iters = 200
batch_size = 32 

# learning hyperparameters
learn_rate = 3e-4
max_iters = 1000
eval_interval = 500

# encoding
end_token = "<END>"
# tokenizer_pattern = "|\d|\\w+|[^\\s]"
# tokenizer_pattern = "|."
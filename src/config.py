# model hyperparameters
block_size = 128 # 256
embed_size = 128  # 64
dropout = 0.2
n_heads = 6
n_layer = 6
eval_iters = 200
batch_size = 64  # 32

# learning hyperparameters
learn_rate = 3e-4
max_iters = 1_000
eval_interval = 500

# encoding
end_token = "<END>"
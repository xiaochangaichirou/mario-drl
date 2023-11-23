import torch as t

lr = 1e-5
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
batch_size = 256
buffer_capacity = 1e6


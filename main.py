import os
from time import time
from neural import Network
from dataset import Dataset

SAVE_FILE = 'model.safetensors'

dataset = Dataset('Iris.csv')

alpha = 1e-5
network = Network([
    4, 16, 64, 256, 3
])

train_dataset = [(o.network_input, o.network_output) for o in dataset.data_objects]

def train():
    global network

    if os.path.exists(SAVE_FILE):
        network.load(SAVE_FILE)
    else:
        network.train(train_dataset, alpha, 600)
        network.save(SAVE_FILE)
    for inp, _ in train_dataset:
        res = network.forward(inp)
        print([float(x) for x in list(res[len(res) - 1])])

def bench():
    global network
    start_time = time()
    for _ in range(0, 1000):
        network.initialize_layer_matrix()
    print(f'Initialize avg time: {(time() - start_time):.6f}ms')

    start_time = time()
    for _ in range(0, 1000):
        network.get_u_mat(train_dataset[0][0])
    print(f'Initialize get_u_mat time: {(time() - start_time):.6f}ms')

    u = network.get_u_mat(train_dataset[0][0])
    start_time = time()
    for _ in range(0, 1000):
        network.forward_u(u)
    print(f'Initialize forward_u time: {(time() - start_time):.6f}ms')

train()
# bench()
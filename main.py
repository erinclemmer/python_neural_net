import os
import torch
from time import time
from neural import Network
from dataset import Dataset

SAVE_FILE = 'model_relu.safetensors'

dataset = Dataset('Iris.csv')

# relu = lambda t: torch.tensor([max(0, x) for x in t]).to('cuda:0')
relu = torch.nn.ReLU()
# relu_derivative = lambda t: torch.tensor([1 if x > 0 else 0 for x in t]).to('cuda:0')
relu_derivative = lambda t: (t > 0).float()

alpha = 1e-5
network = Network([
    4, 16, 64, 256, 3
], relu, relu_derivative)

train_dataset = [(o.network_input, o.network_output) for o in dataset.data_objects]

def train():
    global network

    if os.path.exists(SAVE_FILE):
        network.load(SAVE_FILE)
    else:
        network.train(train_dataset, alpha, 100)
        network.save(SAVE_FILE)
    
    network.train(train_dataset, alpha, 100)

    # for inp, _ in train_dataset:
    #     res = network.forward(inp)
    #     print([float(x) for x in list(res[len(res) - 1])])

def bench():
    global network
    start_time = time()
    for _ in range(0, 1000):
        network.initialize_layer_matrix()
    print(f'Initialize avg time: {(time() - start_time):.6f}ms')

    start_time = time()
    for _ in range(0, 1000):
        network.get_output_mat(train_dataset[0][0])
    print(f'Initialize get_u_mat time: {(time() - start_time):.6f}ms')

    u = network.get_output_mat(train_dataset[0][0])
    start_time = time()
    for _ in range(0, 1000):
        network.activate_output(u)
    print(f'Initialize forward_u time: {(time() - start_time):.6f}ms')

# network.train(train_dataset, alpha, 100)
# network.save(SAVE_FILE)

train()
# bench()
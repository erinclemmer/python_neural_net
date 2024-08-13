import math
from typing import List, Callable

import torch

class Neuron:
    value: float
    weights: List[float]

    def __init__(self, weights: List[float]):
        self.weights = weights

class Layer:
    neurons: List[Neuron]
    last_layer: 'Layer'

    def __init__(self, weights: List[List[float]], last_layer: 'Layer'):
        self.neurons = []
        for n in weights:
            self.neurons.append(Neuron(n))
        self.last_layer = last_layer

lambda_u: Callable[[torch.tensor, torch.tensor], float] = lambda w, x: torch.dot(w, x)

sigmoid: Callable[[float], float] = lambda v: 1 / (1 + math.exp(-v))

lambda_z: Callable[[torch.tensor], torch.tensor] = lambda x: x[len(x) - 1]

lambda_x_backprop: Callable[[float, torch.tensor], torch.tensor] = lambda u, w: (u / torch.dot(w, w)) * w

class Network:
    layer_size: int
    num_layers: int
    weights: List[torch.tensor]

    def __init__(self, num_layers: int, layer_size: int):
        self.layer_size = layer_size
        self.num_layers = num_layers

        # num layers / size of layer / connection to each weight beneath it
        self.weights = torch.rand(num_layers, layer_size, layer_size)
        self.weights[0] = torch.zeros(layer_size, layer_size) # Stays zero because there are no lower neurons
    
    def get_u_mat(self, x: torch.tensor):
        if len(x.size()) > 1 or x.size()[0] != self.layer_size:
            raise f"Input length not correct: found {x.size()} expected {self.layer_size}"

        u_mat = torch.zeros(self.num_layers, self.layer_size)
        u_mat[0] = x
        u = u_mat
        # Per layer
        for i in range(1, self.num_layers):
            # Per neuron
            for j in range(0, self.layer_size):
                u[i][j] = torch.dot(self.weights[i][j], u[i - 1])
        return u

    def forward(self, x: torch.tensor):
        if len(x.size()) > 1 or x.size()[0] != self.layer_size:
            raise f"Input length not correct: found {x.size()} expected {self.layer_size}"

        u = self.get_u_mat(x)
        x = torch.zeros(self.num_layers, self.layer_size)
        for i in range(0, self.num_layers):
            for j in range(0, self.layer_size):
                x[i][j] = sigmoid(u[i][j])

        return x
    
    def loss(self, x: torch.tensor, y: torch.tensor):
        if len(y.size()) > 1 or y.size()[0] != self.layer_size:
            raise f"Output length not correct: found {y.size()} expected {self.layer_size}"
        z = lambda_z(self.forward(x))
        l = y - z
        return (l ** 2) / 2
    
    def backwards(self, x: torch.tensor, y: torch.tensor, alpha: float):
        u = self.get_u_mat(x)
        deltas = torch.zeros(self.num_layers, self.layer_size)
        deltas[self.num_layers - 1] = y
        rev_lyrs = list(range(1, self.num_layers - 1))
        rev_lyrs.reverse()
        for i in rev_lyrs:
            for j in range(0, self.layer_size):
                x: torch.tensor = lambda_x_backprop(u[i][j], self.weights[i][j])
                e: float = torch.dot(self.weights[i][j], x)
                e = sigmoid(e)        
                deltas[i][j] = alpha * e * x
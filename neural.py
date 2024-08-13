import math
from tqdm import tqdm
from typing import List, Callable, Tuple

import torch
from safetensors.torch import save_file, load_file

lambda_u: Callable[[torch.tensor, torch.tensor], float] = lambda w, x: torch.dot(w, x)

sigmoid: Callable[[torch.tensor], torch.tensor] = lambda v: 1 / (1 + math.exp(-v))
sig = torch.nn.Sigmoid()
sigmoid_derivative: Callable[[torch.tensor], torch.tensor] = lambda v: sig(v) * (1 - sig(v))

lambda_z: Callable[[torch.tensor], torch.tensor] = lambda x: x[len(x) - 1]

lambda_x_backprop: Callable[[float, torch.tensor], torch.tensor] = lambda u, w: (u / torch.dot(w, w)) * w

loss: Callable[[torch.tensor, torch.tensor], torch.tensor] = lambda z, y: ((y - z) ** 2) / 2
loss_derivative: Callable[[torch.tensor, torch.tensor], torch.tensor] = lambda z, y: y - z

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
    
    def load(self, file_name: str):
        self.weights = load_file(file_name)["weights"]
        self.num_layers = self.weights.size()[0]
        self.layer_size = self.weights.size()[1]

    def save(self, file_name: str):
        save_file({ "weights": self.weights }, file_name)

    def ensure_input_output_size(self, x: torch.tensor):
        if len(x.size()) > 1 or x.size()[0] != self.layer_size:
            raise f"length not correct: found {x.size()} expected {self.layer_size}"

    def get_u_mat(self, x: torch.tensor):
        self.ensure_input_output_size(x)
        u_mat = torch.zeros(self.num_layers, self.layer_size)
        u_mat[0] = x
        u = u_mat
        # Per layer
        for i in range(1, self.num_layers):
            # Per neuron
            for j in range(0, self.layer_size):
                u[i][j] = torch.dot(self.weights[i][j], u[i - 1])
        return u

    def forward_u(self, u: torch.tensor):
        x = torch.zeros(self.num_layers, self.layer_size)
        for i in range(0, self.num_layers):
            for j in range(0, self.layer_size):
                x[i][j] = sigmoid(u[i][j])

        return x

    def forward(self, x: torch.tensor):
        self.ensure_input_output_size(x)
        u = self.get_u_mat(x)
        return self.forward_u(u)
    
    def predict(self, x: torch.tensor):
        z = self.forward(x)
        z_k = z[self.num_layers - 1]
        max_idx = max(range(len(z_k)), key=z_k.__getitem__)
        return max_idx

    def loss(self, x: torch.tensor, y: torch.tensor):
        self.ensure_input_output_size(y)
        z = self.forward(x)[self.num_layers - 1]
        l = y - z
        return (l ** 2) / 2
    
    def error(self, x: torch.tensor, y: torch.tensor):
        self.ensure_input_output_size(y)
        z = self.forward(x)[self.num_layers - 1]
        return z - y

    def backwards(self, x: torch.tensor, y: torch.tensor):
        u = self.get_u_mat(x)
        x = self.forward_u(u)
        z = x[self.num_layers - 1]
        error = y - z
        
        gradients = torch.zeros(self.num_layers, self.layer_size)
        gradients[self.num_layers - 1] = error * sigmoid_derivative(z)
        
        rev_lyrs = list(range(1, self.num_layers - 1))
        rev_lyrs.reverse()
        for i in rev_lyrs:
            weights  = self.weights[i + 1].t()
            grads = gradients[i + 1]
            sig = sigmoid_derivative(u[i])
            gradients[i] = torch.matmul(weights, grads) * sig
        
        gradient_derivative = torch.zeros(self.num_layers, self.layer_size)
        for i in rev_lyrs:
            gradient_derivative[i] = torch.dot(gradients[i], x[i - 1].t())

        return gradient_derivative
    
    def train(self, 
              dataset: List[Tuple[List[float], List[float]]], 
              alpha: float,
              epochs: int = 5
        ):
        for _ in range(epochs):
            for x, y in tqdm(dataset, "Training model"):
                gradient_derivative = self.backwards(x, y)
                for i in range(0, self.num_layers):
                    self.weights[i] = self.weights[i] - alpha * gradient_derivative[i]
        
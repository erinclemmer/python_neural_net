import math
from tqdm import tqdm
from typing import List, Callable, Tuple

import torch
from safetensors.torch import save_file, load_file

lambda_u: Callable[[torch.tensor, torch.tensor], float] = lambda w, x: torch.dot(w, x)

sigmoid = torch.nn.Sigmoid()
sigmoid_derivative: Callable[[torch.tensor], torch.tensor] = lambda v: sigmoid(v) * (1 - sigmoid(v))

lambda_z: Callable[[torch.tensor], torch.tensor] = lambda x: x[len(x) - 1]

lambda_x_backprop: Callable[[float, torch.tensor], torch.tensor] = lambda u, w: (u / torch.dot(w, w)) * w

loss: Callable[[torch.tensor, torch.tensor], torch.tensor] = lambda z, y: ((y - z) ** 2) / 2
loss_derivative: Callable[[torch.tensor, torch.tensor], torch.tensor] = lambda z, y: y - z

get_max_index: Callable[[List[float]], int] = lambda a: max(range(len(a)), key=a.__getitem__)

class Network:
    num_layers: int
    weights: List[torch.tensor]

    def __init__(self, layers: List[float]):
        self.num_layers = len(layers)
        self.input_size = layers[0]
        self.output_size = layers[self.num_layers - 1]
        self.weights = [torch.zeros(layers[0], 0).to('cuda:0')]
        self.biases = [torch.zeros(layers[0], 0).to('cuda:0')]
        for i in range(1, len(layers)):
            num_neurons = layers[i]
            last_num_layers = layers[i - 1] if i > 0 else 0
            self.weights.append(torch.rand(num_neurons, last_num_layers).to('cuda:0') / 10)
            self.biases.append(torch.rand(num_neurons).to('cuda:0') / 10)
    
    def load(self, file_name: str):
        self.weights = load_file(file_name)["weights"]
        self.num_layers = self.weights.size()[0]

    def save(self, file_name: str):
        save_file({ "weights": self.weights }, file_name)

    def ensure_input_size(self, x: torch.tensor):
        if len(x.size()) > 1 or x.size()[0] != self.input_size:
            raise f"Input length not correct: found {x.size()} expected {self.input_size}"
        
    def ensure_output_size(self, y: torch.tensor):
        if len(y.size()) > 1 or y.size()[0] != self.output_size:
            raise f"Output length not correct: found {y.size()} expected {self.output_size}"

    def initialize_layer_matrix(self):
        layers = []
        for lyr in self.weights:
            layers.append(torch.zeros(lyr.size()[0]).to('cuda:0'))
        return layers

    def get_u_mat(self, x: torch.tensor):
        x = x.to('cuda:0')
        self.ensure_input_size(x)
        u = self.initialize_layer_matrix()
        u[0] = x
        # Per layer
        for i in range(1, self.num_layers):
            # Per neuron
            for j in range(0, self.weights[i].size()[0]):
                u[i][j] = torch.dot(self.weights[i][j], u[i - 1]) + self.biases[i][j]
        return u

    def forward_u(self, u: torch.tensor):
        x = self.initialize_layer_matrix()
        for i in range(0, self.num_layers):
            for j in range(0, x[i].size()[0]):
                x[i][j] = sigmoid(u[i][j])

        return x

    def forward(self, x: torch.tensor):
        x = x.to('cuda:0')
        self.ensure_input_size(x)
        u = self.get_u_mat(x)
        return self.forward_u(u)
    
    def predict(self, x: torch.tensor):
        x = x.to('cuda:0')
        z = self.forward(x)
        z_k = z[self.num_layers - 1]
        max_idx = get_max_index(z_k)
        return max_idx

    def loss(self, x: torch.tensor, y: torch.tensor):
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        self.ensure_output_size(y)
        z = self.forward(x)[self.num_layers - 1]
        l = y - z
        return (l ** 2) / 2
    
    def error(self, x: torch.tensor, y: torch.tensor):
        self.ensure_output_size(y)
        z = self.forward(x)[self.num_layers - 1]
        return z - y

    def backwards(self, x: torch.tensor, y: torch.tensor):
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        u = self.get_u_mat(x)
        x = self.forward_u(u)
        z = x[self.num_layers - 1]
        error = y - z
        
        gradients = self.initialize_layer_matrix()
        gradient_derivative = self.initialize_layer_matrix()
        gradients[self.num_layers - 1] = error * sigmoid_derivative(z)
        
        for i in range(self.num_layers - 1, 0, -1):
            if i < self.num_layers - 1:
                weights = self.weights[i + 1].t()
                grads = gradients[i + 1]
                gradients[i] = torch.matmul(weights, grads) * sigmoid_derivative(u[i])
            gradient_derivative[i] = torch.outer(gradients[i], x[i - 1])

        return gradient_derivative, gradients
    
    def train(self, 
              dataset: List[Tuple[List[float], List[float]]], 
              alpha: float,
              epochs: int = 5
        ):
        for i in range(epochs):
            print(f'Epoch: {i}')
            ttl_loss = 0
            for x, y in tqdm(dataset, "Training model"):
                gradient_derivative, gradients = self.backwards(x, y)
                avg_item_loss = 0
                for l in list(self.loss(x, y)):
                    avg_item_loss += l
                ttl_loss += avg_item_loss / y.size()[0]
                for j in range(1, self.num_layers):
                    self.weights[j] -= alpha * gradient_derivative[j]
                    self.biases[j] -= alpha * gradients[j]
            print(f'Avg loss: {ttl_loss / len(dataset):.4f}')
            score = 0
            for x, y in dataset:
                prediction = self.predict(x)
                truth = get_max_index(y)
                if prediction == truth:
                    score += 1
            print(f'Percent Correct: {score / len(dataset) * 100:.2f}%')
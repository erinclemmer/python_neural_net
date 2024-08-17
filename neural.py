import os
import re
import json
from time import time
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

    def __init__(
            self, 
            layers: List[float], 
            activation: Callable[[torch.tensor], torch.tensor] = sigmoid, 
            activation_derivative: Callable[[torch.tensor], torch.tensor] = sigmoid_derivative
        ):
        self.num_layers = len(layers)
        self.input_size = layers[0]
        self.output_size = layers[self.num_layers - 1]
        self.weights = [torch.zeros(layers[0], 0).to('cuda:0')]
        self.biases = [torch.zeros(layers[0], 0).to('cuda:0')]
        self.activation = activation
        self.activation_derivative = activation_derivative
        for i in range(1, len(layers)):
            num_neurons = layers[i]
            last_num_layers = layers[i - 1] if i > 0 else 0
            self.weights.append(torch.rand(num_neurons, last_num_layers).to('cuda:0') / 10)
            self.biases.append(torch.rand(num_neurons).to('cuda:0') / 10)
    
    def load(self, file_name: str):
        data = load_file(file_name)
        self.weights = [None for _ in range(0, len(data.keys()) // 2)]
        self.biases = [None for _ in range(0, len(data.keys()) // 2)]
        reg = re.compile('layer_([0-9]+)')
        for key in data.keys():
            layer = int(reg.findall(key)[0])
            if 'weights' in key:
                self.weights[layer] = data[key].to('cuda:0')
            if 'bias' in key:
                self.biases[layer] = data[key].to('cuda:0')

    def save(self, file_name: str):
        o = { }
        for lyr in range(0, len(self.weights)):
            o[f'weights_layer_{lyr}'] = self.weights[lyr]
            o[f'bias_layer_{lyr}'] = self.biases[lyr]
        save_file(o, file_name)

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
            u[i] = torch.matmul(self.weights[i], u[i - 1]) + self.biases[i]
        return u

    def forward_u(self, u: torch.tensor):
        x = self.initialize_layer_matrix()
        for i in range(0, self.num_layers):
            x[i] = self.activation(u[i])

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

    def loss_z(self, z: torch.tensor, y: torch.tensor):
        y = y.to('cuda:0')
        z = z.to('cuda:0')
        self.ensure_output_size(y)
        self.ensure_output_size(z)
        l = y - z
        return (l ** 2) / 2

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

    def backwards_new(self, inp: torch.tensor, truth: torch.tensor):
        inp = inp.to('cuda:0')
        truth = truth.to('cuda:0')
        outp = self.get_u_mat(inp)
        y_hat = outp[self.num_layers - 1]
        activations = self.forward_u(outp)
        loss_derivative = (y_hat - truth)
        activation_derivative = [self.activation_derivative(lyr) for lyr in outp]

        # k
        last_layer = self.num_layers - 1

        # assume nxm matrices for weights and 1xn vectors for bias
        delta = self.initialize_layer_matrix()
        grad_weights = self.initialize_layer_matrix()
        grad_bias = self.initialize_layer_matrix()

        # diminsionality for surrounding layers
        # weights[k]    = dn
        # weights[k-1]  = dm

        # Derivative of loss w.r.t last layer bias
        # dL/db^k   = dL/dy^ * dy^/dz[k] * dz[k]/db[k] 
        #           = (y^ - y) * sigma prime (z[k]) * 1
        # element wise multiplication
        grad_bias[last_layer] = loss_derivative * activation_derivative[last_layer]
        
        # Intermediate value for saving computation
        # delta^k   = dL/da[k-1] = dL/dy^ * dy^/dz[k] * dz[k]/da[k-1] 
        #           = (y^ - y) * sigma prime (z[k]) * W[k]
        # mat mul: 1 x dn * dn x dm = 1 x dm
        delta[last_layer] = grad_bias[last_layer] @ self.weights[last_layer]
        
        # Derivative of loss w.r.t. last layer weights
        # dL/W^k    = dL/dy^ * dy^/dz[k] * dz[k]/dW[k] 
        #           = dL/dz^k * dz[k]/dW[k] 
        #           = (y^ - y) * sigma prime (z[k]) * a[k-1]
        # outer prod: 1 x dm   1 x dn = dn x dm
        grad_weights[last_layer] = torch.outer(delta[last_layer], activations[last_layer - 1])
        
        for i in range(1, self.num_layers - 1):
            # c = k - n
            current_layer = last_layer - i
            
            # diminsionality for surrounding layers
            # layer[c+1]    = dn
            # layer[c]      = dm
            # layer[c-1]    = do
            
            # derivative of loss w.r.t bias at layer k - n
            # dL/db[c]    = dL/dy^ * dy^/dz[k] * { PI (i: 0 -> n - 1) dz/[k - i] } * { PI (i: 1 -> n) da[k-i]/dz[k-i] } * dz[c]/db[c]
            #               = dL/db[c+1] * dz[c]/da[c] * da[c]/db[c]
            #               = dL/db[c+1] * W[c] * sigma prime (z[c]) * 1
            # dL/db[c+1]: 1 x dn, W[c]: dm x do, sigma prime (z[c]): 1 x dm
            grad_bias[current_layer] = grad_bias[current_layer + 1] @ self.weights[current_layer] * activation_derivative[current_layer]

            # delta[c] = dL/da[c]
            delta[current_layer] = delta[current_layer + 1] * self.weights[current_layer] * activation_derivative[current_layer]

            # derivative of loss w.r.t weights at layer k - n
            # dL/dW[c]    = dL/dy^ * dy^/dz[k] * { PI (i: 0 -> n - 1) dz[k-i]/da[k-i-1] } * { PI (i: 1 -> n ) da[k-i]/dz[k-i] }  * dz[c]/dW[c]
            #               = dL/a[c+1]
            grad_weights[current_layer] = grad_bias[current_layer] * activations[current_layer - 1]
        
        loss = (loss_derivative ** 2) / 2

        return grad_weights, grad_bias, loss

        

    def backwards(self, x: torch.tensor, y: torch.tensor):
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        u = self.get_u_mat(x)
        x = self.forward_u(u)
        z = x[self.num_layers - 1]
        error = y - z
        
        gradients = self.initialize_layer_matrix()
        gradient_derivative = self.initialize_layer_matrix()
        gradients[self.num_layers - 1] = error * self.activation_derivative(z)
        
        for i in range(self.num_layers - 1, 0, -1):
            if i < self.num_layers - 1:
                weights = self.weights[i + 1].t()
                grads = gradients[i + 1]
                gradients[i] = torch.matmul(weights, grads) * self.activation_derivative(u[i])
            gradient_derivative[i] = torch.outer(gradients[i], x[i - 1])

        loss = self.loss_z(z, y)

        return gradient_derivative, gradients, loss
    
    def train(self, 
              dataset: List[Tuple[List[float], List[float]]], 
              alpha: float,
              epochs: int = 5
        ):

        def test_model():
            score = 0
            for x, y in dataset:
                prediction = self.predict(x)
                truth = get_max_index(y)
                if prediction == truth:
                    score += 1
            print(f'Percent Correct: {score / len(dataset) * 100:.2f}%')

        def compute_pair(x: torch.tensor, y: torch.tensor):
            gradient_derivative, gradients, loss = self.backwards_new(x, y)
            avg_item_loss = 0
            for l in list(loss):
                avg_item_loss += l
            for j in range(1, self.num_layers):
                self.weights[j] += alpha * gradient_derivative[j]
                self.biases[j] += alpha * gradients[j]
            return avg_item_loss / y.size()[0]

        def do_epoch():
            # print(f'Epoch: {i}')
            ttl_loss = 0
            start_time = time()
            for x, y in dataset:
                ttl_loss += compute_pair(x, y)
            # print(f'Avg backwards time: {(time() - start_time) / len(dataset) * 1000:.4f}ms')
            avg_loss = ttl_loss / len(dataset)
            # print(f'Avg loss: {avg_loss:.4f}')
            return float(avg_loss)

        losses = []
        for _ in tqdm(range(epochs), 'Epoch'):
            losses.append(do_epoch())
        if not os.path.exists('results'):
            os.mkdir('results')
        with open(f'results/training_stats_a{alpha}_e{epochs}.json', 'w', encoding='utf-8') as f:
                json.dump({
                    "network": [len(lyr) for lyr in self.weights],
                    "alpha": alpha,
                    "epochs": epochs,
                    "losses": losses
                }, f, indent=4)
        test_model()
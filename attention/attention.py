import torch

class OutputLayer:
    weights: torch.tensor

    def __init__(self, w: torch.tensor):
        self.weights = w
        self.sigma = torch.nn.Softmax(dim=1)
        self.sigma_predict = torch.nn.Softmax(dim=0)

    def logits(self, x: torch.tensor):
        return x @ self.weights

    def forward(self, x: torch.tensor):
        z = self.logits(x)
        return self.sigma(z)
    
    def predict(self, x: torch.tensor):
        z = self.logits(x)
        return torch.argmax(self.sigma_predict(z[-1]))
    
    def backwards(self, x: torch.tensor, y: torch.tensor):
        y_hat = self.forward(x)
        diff = y_hat - y
        return diff @ x.t()
    
    
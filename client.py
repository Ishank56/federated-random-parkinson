from collections import OrderedDict
import torch
import flwr as fl
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from model import ParkinsonsNet, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, input_dim):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = ParkinsonsNet(input_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar] = None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config['lr']
        epochs = config['local_epochs']
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.valloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloader, valloader, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloader, valloader=valloader, input_dim=input_dim)
    return client_fn
import flwr as fl
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from model import FederatedRandomForest, train, test
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import pickle

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, input_dim):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = FederatedRandomForest(input_dim)

    def get_parameters(self, config: Dict[str, Scalar] = None):
        # Return model parameters as a list of NumPy ndarrays
        model_bytes = self.model.get_params()
        return [np.frombuffer(model_bytes, dtype=np.uint8)]

    def set_parameters(self, parameters: NDArrays):
        # Set model parameters from a list of NumPy ndarrays
        model_bytes = parameters[0].tobytes()
        self.model.set_params(model_bytes)

    def fit(self, parameters, config):
        # Set model parameters, train model, return updated parameters
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        model_bytes = self.model.get_params()
        return [np.frombuffer(model_bytes, dtype=np.uint8)], len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Set model parameters, evaluate model on test data, return loss and accuracy
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloader, valloader, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloader, valloader=valloader, input_dim=input_dim)
    return client_fn

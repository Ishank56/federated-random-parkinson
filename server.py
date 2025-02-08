from omegaconf import DictConfig
from typing import List
import flwr as fl
from model import FederatedRandomForest, test
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import NDArrays, Scalar
import pickle

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {'local_epochs': config.local_epochs}
    return fit_config_fn

def get_evaluate_fn(input_dim, test_loader):
    def evaluate_fn(server_round: int, parameters: NDArrays, config: dict):
        # Initialize a new Random Forest model
        model = FederatedRandomForest(input_dim)

        # Set the parameters to the model
        model_bytes = parameters[0].tobytes()
        model.set_params(model_bytes)

        # Fit the model on the test data (or a portion of it)
        X_test = test_loader.dataset.tensors[0].numpy()
        y_test = test_loader.dataset.tensors[1].numpy()
        model.fit(X_test, y_test)
    
        # Evaluate the model
        loss, accuracy = test(model, test_loader)
        
        return loss, {"accuracy": accuracy}
    return evaluate_fn

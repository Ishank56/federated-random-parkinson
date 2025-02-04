from omegaconf import DictConfig
from collections import OrderedDict
import torch
import flwr as fl
from model import ParkinsonsNet, test
from typing import List

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {'lr': config.lr, 'momentum': config.momentum, 'local_epochs': config.local_epochs}
    return fit_config_fn

def get_evaluate_fn(input_dim, test_loader):
    def evaluate_fn(server_round: int, parameters: List[float], config: dict):
        model = ParkinsonsNet(input_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert Flower parameters to PyTorch state_dict
        state_dict = {k: torch.tensor(v).to(device) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        loss, accuracy = test(model, test_loader, device)
        return loss, {"accuracy": accuracy}
    return evaluate_fn
from collections import OrderedDict
import torch
import tensorflow as tf
import flwr as fl
from typing import Dict,Tuple
from flwr.common import NDArrays, Scalar

from model import Net, test, train

 # If Scalar is meant to be a numpy scalar


# Corrected model definition
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                trainloader,
                valloader,
                num_classes)-> None:
        super().__init__()

        self.trainloader=trainloader
        self.valloader=valloader

        self.model=Net(num_classes)

        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):

        params_dict= zip(self.model.state_dict().keys(), parameters)

        state_dict=OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)
        # return model.get_weights()

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):

        #copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr=config['lr']
        momentum=config['momentum']
        epochs=config['local_epochs']

        optim=torch.optim.SGD(self.model.parameters(), lr=lr ,momentum=momentum)

        #Do local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters:NDArrays, config:Dict[str, Scalar]):

        self.set_parameters(parameters)

        loss, accuracy= test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy':accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes):


    def client_fn(cid: str):

        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            )
    
    return client_fn

    




        # model.fit(x_train, y_train, epochs=1, batch_size=32)
        # return model.get_weights(), len(x_train), {}

    # def evaluate(self, parameters, config):
    #     model.set_weights(parameters)
    #     loss, accuracy = model.evaluate(x_test, y_test)
    #     return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
# fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient)

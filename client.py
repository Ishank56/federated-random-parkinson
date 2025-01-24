# If Scalar is meant to be a numpy scalar


# Corrected model definition
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

import tensorflow as tf
import flwr as fl
from typing import Dict
from model import Net

class FlowerClient(fl.client.NumPyClient):
    def _init_(self, trainloader, valloader, input_shape):
        super()._init_()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = create_cnn_model(input_shape)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def get_parameters(self):
        return [val.numpy() for val in self.model.trainable_weights]

    def set_parameters(self, parameters):
        for var, param in zip(self.model.trainable_weights, parameters):
            var.assign(param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.trainloader, epochs=config['local_epochs'])
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.valloader)
        return loss, len(self.valloader), {"accuracy": accuracy}


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

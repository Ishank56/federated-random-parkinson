import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
import pickle

class FederatedRandomForest:
    def __init__(self, input_dim):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters here
        self.input_dim = input_dim

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return pickle.dumps(self.model)

    def set_params(self, params):
        self.model = pickle.loads(params)

def train(model, train_loader):
    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    model.fit(X_train, y_train)

def test(model, test_loader):
    X_test = test_loader.dataset.tensors[0].numpy()
    y_test = test_loader.dataset.tensors[1].numpy()
    y_pred = model.predict(X_test)
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    return loss, accuracy

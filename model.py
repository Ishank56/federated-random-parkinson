import pandas as pd
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class Net(nn.Module):

    def create_cnn_model(input_shape):
        model = models.Sequential()
        model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(64, 3, activation='relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(2, activation='softmax'))
        return model

# Load and preprocess dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Assuming the last column is the target (class labels), modify if different.
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    labels = to_categorical(labels)  # Convert labels to one-hot encoding
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# Train function
def train(model, train_data, train_labels, optimizer, loss_fn, epochs=10, batch_size=32):
    """Train the model."""
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")

# Test function
def test(model, test_data, test_labels, loss_fn):
    """Evaluate the model."""
    dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    loss, accuracy = 0.0, 0.0
    num_batches = 0
    for x_batch, y_batch in dataset:
        predictions = model(x_batch, training=False)
        batch_loss = loss_fn(y_batch, predictions)
        loss += batch_loss.numpy()
        accuracy += np.sum(np.argmax(predictions.numpy(), axis=1) == np.argmax(y_batch, axis=1))
        num_batches += 1
    loss /= num_batches
    accuracy /= len(test_data)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return loss, accuracy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def prepare_dataset(num_partitions, batch_size, data_path="/home/tt603/Desktop/federated_learning/Federated-Framework/Parkinsson disease.csv"):
    data = pd.read_csv(data_path).drop(columns=['name'])
    X = data.drop(columns=['status']).values
    y = data['status'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    partition_size = len(X_scaled) // num_partitions
    trainloaders, valloaders = [], []

    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        X_partition = X_scaled[start_idx:end_idx]
        y_partition = y[start_idx:end_idx]

        if len(X_partition) < 2:  # Ensure enough samples in the partition
            print(f"Skipping partition {i} due to insufficient samples.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X_partition, y_partition, test_size=0.1, random_state=42, stratify=y_partition
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Skipping partition {i} due to insufficient train/validation samples.")
            continue

        X_train_cnn = X_train[..., np.newaxis]
        X_val_cnn = X_val[..., np.newaxis]

        trainloader = tf.data.Dataset.from_tensor_slices((X_train_cnn, y_train)).batch(batch_size)
        valloader = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val)).batch(batch_size)

        trainloaders.append(trainloader)
        valloaders.append(valloader)

    X_test, _, y_test, _ = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)
    X_test_cnn = X_test[..., np.newaxis]
    testloader = tf.data.Dataset.from_tensor_slices((X_test_cnn, y_test)).batch(batch_size)

    return trainloaders, valloaders, testloader
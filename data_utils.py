import torch
import torchvision
from torchvision.datasets import mnist

# Splitting the data before sending to the clients
# Returns list of torch.data.utils.Subset with data for each client + validation set (the last element in the list)
def get_partitions(N_clients):
    # Load mnist
    mnist_data=torchvision.datasets.MNIST('./data',download=True,transform=torchvision.transforms.ToTensor())

    # Calculate the size of the training and validation sets
    len_mnist=len(mnist_data)
    len_test=int(len_mnist * 0.2)
    len_train=len_mnist - len_test

    # Calculate the size of the training set for each client
    partition_size=len_train // N_clients

    # Create a list of all partition lengths. This list will be [client1_len, client2_len, ..., clientN_len, val_len]
    lengths=[partition_size]*N_clients + [len_test]

    # Handle the remainder. If len_train is not divisible by N_clients, then add residual data to the first client
    if (partition_size * N_clients < len_train):
        lengths[0]+=len_mnist-len_test-partition_size*N_clients

    # Split the data
    partitions=torch.utils.data.random_split(mnist_data,lengths)

    return partitions
import torch
import torchvision
from torchvision.datasets import mnist

# For splitting the data before sending to the clients
def get_partitions(N_clients):
    # Load mnist
    mnist_data=torchvision.datasets.MNIST('./data')
    partition_size=len(mnist)//N_clients
    lengths=[partition_size]*N_clients
    # If len(mnist) is not divisible by N_clients, then add residual data to the first client
    if (partition_size*N_clients < len(mnist)):
        lengths[0]+=len(mnist)-partition_size*N_clients
    # Split the data
    partitions=torch.utils.data.random_split(mnist_data,lengths)
    return partitions
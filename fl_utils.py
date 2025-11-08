import torch
import torch.nn as nn

from collections import OrderedDict

def federated_average(list_of_state_dicts):

    if not list_of_state_dicts:
        return None
    
    keys = list_of_state_dicts[0].keys()

    avg_state_dict = OrderedDict()

    for key in keys:
        layer_tensors = torch.stack([sd[key] for sd in list_of_state_dicts])

        layer_avg = torch.mean(layer_tensors, dim=0)

        avg_state_dict[key] = layer_avg

    return avg_state_dict

# Local training of the model using mini-batches
def train_local(model,images,labels,numEpochs,learningRate):
    # Define tain loader, loss function and optimizer
    train_loader=torch.utils.data.DataLoader(images,batch_size=64,shuffle=True)
    lossFun=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate)

    #Training
    for i in range(numEpochs):
        for images, labels in train_loader:
            # Forward step
            pred=model(images)

            # Compute loss
            loss=lossFun(labels,pred)

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Testing results using torchmetrics.Accuracy
def test_global(preds, labels):
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        return accuracy(preds, labels)
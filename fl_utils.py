import torch
import torch.nn as nn
import torchmetrics

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
def train_local(model,data,numEpochs,learningRate):
    # Define tain loader, loss function and optimizer
    train_loader=torch.utils.data.DataLoader(data,batch_size=64,shuffle=True)
    lossFun=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)

    #Training
    for i in range(numEpochs):
        for images, labels in train_loader:
            # Forward step
            pred=model(images)
            # Compute loss
            loss=lossFun(pred,labels)

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Testing results using torchmetrics.Accuracy
def test_global(model,test_data):
    test_loader=torch.utils.data.DataLoader(test_data,shuffle=True)
    accuracy=torchmetrics.Accuracy(task='multiclass',num_classes=10)
    for image, label in test_loader:
        pred=model(image)
        accuracy.update(pred,label)
    final_accuracy=accuracy.compute()
    return final_accuracy.item()
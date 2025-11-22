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

# Evaluates global model performance in batches. Runs after each training round.
def test_global(model,test_data):
    batch_size = 128
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)
    total_loss = 0 # total loss computed sum of loss * batch size for each batch
    correct = 0 # number of correct predicted samples
    total_samples = 0
    lossFun = nn.CrossEntropyLoss()

    # Starting batch testing
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            yHat = model(X)
            loss = lossFun(yHat, y)
            total_loss += loss.item() * X.size(0)
            predicted = yHat.argmax(1)
            total_samples += y.size(0)
            correct += (predicted == y).sum().item()

    # Compute average loss (loss per sample)
    avg_loss = total_loss / total_samples

    # Compute accuracy in %
    accuracy = 100 * correct / total_samples
    return avg_loss, accuracy
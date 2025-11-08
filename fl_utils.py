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
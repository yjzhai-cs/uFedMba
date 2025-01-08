import copy
import torch
from torch import nn


def aggregate_parameters(params_list, weights):
    aggregated_params = []

    for params in zip(*params_list):
        aggregated_params.append(
            torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
        )
    
    return aggregated_params

def personalize_parameters(parameters, gamma_matrix, clients_parameters_list):
    personalized_parameters = []

    layer = 0
    for params in zip(*clients_parameters_list):
        personalized_parameters.append(
            0.5 * parameters[layer] + 0.5 * torch.sum(gamma_matrix[layer] * torch.stack(params, dim=-1), dim=-1)
        )
        layer += 1
    
    return personalized_parameters
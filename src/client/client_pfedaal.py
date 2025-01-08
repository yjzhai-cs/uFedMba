import numpy as np
import copy 

from collections import OrderedDict
from typing import OrderedDict

import torch 
from torch import nn, optim
import torch.nn.functional as F

def get_based_and_personalzied_layers_name(model: torch.nn.Module, L: int):
    layers_name = [n for n, _ in model.state_dict().items()]
    layers_num = len(layers_name)
    
    if layers_num < L:
        raise RuntimeError(f"layers_num < L({L})")
    
    based_layers_name = layers_name[:layers_num - L]
    personalized_layers_name = layers_name[layers_num - L:]
    
    return based_layers_name, personalized_layers_name

def compute_cos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    a = A.flatten()
    b = B.flatten()

    return torch.sum(a * b) / (torch.norm(a) * torch.norm(b) + 1e-12)

class Client_pFedAAL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, L, mu,
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device 
        self.L = L
        self.mu = mu
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        self.based_layers_name, self.personalized_layers_name = get_based_and_personalzied_layers_name(self.net, self.L)
        
    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        original_parameters = copy.deepcopy(self.net.state_dict())

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)

                fed_prox_reg = 0.0
                for param_name, param in self.net.state_dict().items():
                    if param_name in self.based_layers_name:
                        fed_prox_reg += ((self.mu / 2) * torch.norm((param - original_parameters[param_name]))**2)
                
                # print("fed_prox_reg: ")
                # print(fed_prox_reg)
                loss += fed_prox_reg

                loss.backward() 
                        
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        delta = OrderedDict(
            {
                k1: p1 - p0
                for (k1, p1), (k0, p0) in zip(
                    self.net.state_dict().items(),
                    original_parameters.items(),
                )
            }
        )

#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 
        
        return delta, sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self):
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy

import torch
import torch.nn as nn
from rrelu.relu_bound.bound_relu import Relu_bound
class bounded_relu_tresh(nn.Module,Relu_bound):
    '''
    Bound the relu activatoin and back the values to treshold
    ------------------------------------
    bound : the bound for the activation
    -------------------------------------
    pytorch module with forward function
    '''
    def __init__(self, bounds ,tresh=None,alpha=None,k=-1):
        super().__init__()
        self.bounds = bounds
        self.tresh = tresh
    def forward(self, input):
        # input = torch.nan_to_num(input)
        output = torch.ones_like(input) * input
        output[torch.gt(input,self.bounds)] = self.tresh
        return torch.maximum(torch.tensor(0.0),output)

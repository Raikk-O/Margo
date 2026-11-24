
import copy

import torch.nn as nn


def replace_act(model: nn.Module, name_relu_bound: str, name_serach_bound: str, data_loader, bound_type: str,
                bitflip: str) -> nn.Module:
    replace_act_dict = {
        'zero': bounded_relu_zero,
        'tresh': bounded_relu_tresh,
        'fitact': bounded_relu_fitact,
        'proact': bounded_hyrelu_proact,
        'rlact': bounded_hyrelu_rlact
    }

    bounds, tresh, alpha = find_bounds(copy.deepcopy(model), data_loader, name_serach_bound, bound_type, bitflip)
    print("from src/rrelu/setup.py")
    print("information:bounds", bounds)
    print("information:tresh", tresh)
    model = replace_act_all(model, replace_act_dict[name_relu_bound], bounds, tresh, alpha, name='')
    return model

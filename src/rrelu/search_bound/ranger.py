import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 初始化全局变量 activation
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def relu_hooks(model: nn.Module, name=''):
    for name1, layer in model.named_children():
        if len(list(layer.children())) == 0:
            if isinstance(layer, nn.ReLU):
                name_ = name1 + name
                layer.register_forward_hook(get_activation(name_))
                print(f"Registered hook on {name_}")
        else:
            relu_hooks(layer, name + name1)

# Ranger_bounds 函数
def Ranger_bounds(model: nn.Module, train_loader, device="cuda", bound_type='layer', bitflip='float'):
    model.eval()
    iteration = True
    results = {}
    tresh = {}

    # print(model, train_loader, device, bound_type, bitflip)
    # 注册钩子函数
    relu_hooks(model, name='')

    for data, label in train_loader['sub_train']:
        data = data.to(device)
        label = label.to(device)
        model = model.to(device)
        output = model(data)

        # print("Output shape:", output.shape)

        if iteration:
            for key, val in activation.items():
                results[key] = val
                tresh[key] = val
            iteration = False

        # print("Activation dictionary after forward pass:")
        # for key, val in activation.items():
        #     print(f"{key}: {val.shape}")

        for key, val in activation.items():
            prev_max = torch.max(results[key], dim=0)[0]
            prev_mean = torch.mean(tresh[key], dim=0)
            curr_max = torch.max(activation[key], dim=0)[0]
            curr_mean = torch.mean(activation[key], dim=0)
            results[key] = torch.maximum(prev_max, curr_max)
            tresh[key] = torch.minimum(prev_mean, curr_mean)

    if bound_type == "layer":
        for key, val in results.items():
            results[key] = torch.max(val)
            tresh[key] = torch.min(tresh[key])

    return results, tresh, None




   


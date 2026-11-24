# -*- coding: utf-8 -*-
"""
replace_act — 将模型中的 nn.ReLU 替换为有界 ReLU(bounded_relu_*)。

说明
----
本文件被 experiments/trajectory/ 下的入口脚本通过 ``_extract_functions`` 提取
``replace_act`` 函数源码后, 在受限命名空间中 ``exec`` 执行。函数体内引用的
``find_bounds`` / ``replace_act_all`` / ``bounded_relu_*`` 均由入口脚本注入,
因此本文件不依赖其它顶层 import。
"""
import copy

import torch.nn as nn


def replace_act(model: nn.Module, name_relu_bound: str, name_serach_bound: str, data_loader, bound_type: str,
                bitflip: str) -> nn.Module:
    # 激活函数字典: 键是激活函数名称, 值是对应实现(由入口脚本注入)
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

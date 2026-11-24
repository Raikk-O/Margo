"""pytorchfi.core contains the core functionality for fault injections"""

import copy
import logging
import warnings
from typing import List

import torch
import torch.nn as nn
from rrelu.relu_bound.bound_relu import Relu_bound

class FaultInjection:
    def __init__(
        self,
        model,
        batch_size: int,
        input_shape: List[int] = None,
        layer_types=None,
        total_bits = 32, 
        n_frac = 16,
        n_int = 15, 
        **kwargs,
    ):
        if not input_shape:
            input_shape = [3, 224, 224]
        if not layer_types:
            layer_types = [nn.Conv2d]
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.original_model = model.cuda()
        self.output_size = []
        self.layers_type = []
        self.layers_dim = []
        self.weights_size = []
        self.bias_size= []
        self.batch_size = batch_size
        self._input_shape = input_shape
        self._inj_layer_types = layer_types
        self.total_bits = total_bits
        self.n_frac = n_frac
        self.n_int = n_int
        self.corrupted_model = None
        self.current_layer = 0
        self.handles = []
        self.corrupt_batch = []
        self. corrupt_layer = []
        # profiling forward 时真正触发 hook 的层(按 forward 执行顺序)及其在
        # 静态收集列表中的位置索引, 用于让注入层与 output_size/layers_type 对齐
        self._hooked_modules = []
        self._hooked_pos = None

        #说明
        # self.corrupt_dim[0]用来存储通道 c 索引
        # self.corrupt_dim[1]用来存储高度 h 索引
        # self.corrupt_dim[2]用来存储宽度 w 索引
        self.corrupt_dim = [[], [], []]  # C, H, W
        self.corrupt_value = []

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)

        if not isinstance(input_shape, list):
            raise AssertionError("Error: Input shape must be provided as a list.")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise AssertionError("Error: Batch size must be an integer greater than 1.")
        if len(layer_types) < 0:
            raise AssertionError("Error: At least one layer type must be selected.")

        handles, _shapes, self.weights_size,self.bias_size = self._traverse_model_set_hooks(
            self.original_model, self._inj_layer_types
        )

        dummy_shape = (1, *self._input_shape)  # profiling only needs one batch element
        model_dtype = next(model.parameters()).dtype
        device = "cuda" if self.use_cuda else None
        _dummy_tensor = torch.randn(dummy_shape, dtype=model_dtype, device=device)

        self._hooked_modules = []
        self.original_model(_dummy_tensor)

        for index, _handle in enumerate(handles):
            handles[index].remove()

        # 只有真正执行了 forward 的层才会触发 hook。按触发顺序记录它们在静态
        # 收集列表中的位置, 供 _collect_inj_layers 使用, 使注入层顺序与
        # self.output_size / self.layers_type 严格一致。
        _static_layers = self._static_collect(self.original_model, self._inj_layer_types)
        _pos_of_id = {}
        for _i, _m in enumerate(_static_layers):
            _pos_of_id.setdefault(id(_m), _i)
        self._hooked_pos = [_pos_of_id[id(m)] for m in self._hooked_modules if id(m) in _pos_of_id]

        # weights_size / bias_size 由 _traverse_model_set_hooks 按静态遍历顺序收集,
        # 与 profiling 得到的 output_size / layers_type(forward 顺序)并不一致。
        # 不同步重排的话, 生成故障位置时会取到别的层的 shape, 把故障注入到错误位置
        # 甚至越界(例如用 4 维索引去访问 Relu_bound 的标量 bounds)。
        if len(self.weights_size) == len(_static_layers) and self._hooked_pos:
            self.weights_size = [self.weights_size[i] for i in self._hooked_pos]
        if len(self.bias_size) == len(_static_layers) and self._hooked_pos:
            self.bias_size = [self.bias_size[i] for i in self._hooked_pos]

        logging.info("Input shape:")
        logging.info(dummy_shape[1:])

        logging.info("Model layer sizes:")
        logging.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.output_size
                ]
            )
        )

    def reset_fault_injection(self):
        self._reset_fault_injection_state()
        self.corrupted_model = None
        logging.info("Fault injector reset.")

    def _reset_fault_injection_state(self):
        (
            self.current_layer,
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
            self.corrupt_value,
        ) = (0, [], [], [[], [], []], [])

        for index, _handle in enumerate(self.handles):
            self.handles[index].remove()

#这段代码的主要目的是在神经网络模型的每一层上注册forward hook 用于在前向传播时捕获输出，
    #权重和偏置的形状（维度）。通过对每一层的遍历，收集与该层相关的信息，包括该层的输出，
    #权重和偏置的形状（尺寸）。以下是逐行解析这段代码的具体实现

    #model是当前传入的模型 layer_types是一个层类型的列表，表示需要设置钩子的层的类型
    def _traverse_model_set_hooks(self, model, layer_types):

        handles = [] #存储每个注册的hook（钩子）的句柄
        output_shape = [] #用于存储每一层的输出形状
        weights_shape = [] #用于存储每一层的权重的形状
        bias_shape =[] #用于存储每一层的偏置形状

        #遍历模型的每一层
        for layer in model.children():
            # leaf node 处理叶节点（没有子层的层）
            if list(layer.children()) == []:
                #检查层类型并注册hook
                #如果layer_types中包含“all”,表示对所有层都设置hook
                #于是使用register_forward_hook注册钩子函数self._save_output_size
                #并且将返回的句柄(handle)到handled列表
                if "all" in layer_types:
                    handles.append(layer.register_forward_hook(self._save_output_size))
                else:
                #如果layer_types中没有“all”,则遍历 layer_types列表，如过当前层是指定的
                #某种类型（例如 Conv2d ,linear等），则注册hook，
                # 并将该层的引用添加到output_shape列表中
                    for i in layer_types:
                        if isinstance(layer, i):

                            # neurons
                            handles.append(
                                layer.register_forward_hook(self._save_output_size)
                            )
                            #添加的是层对象了
                            output_shape.append(layer)

                            # weights
                            if isinstance(layer,Relu_bound):
                                if len(layer.bounds.shape) == 0 : 
                                    weights_shape.append([0])
                                else:
                                    weights_shape.append(layer.bounds.shape)    
                                if layer.tresh!=None:
                                    if len(layer.tresh.shape)==0:
                                        bias_shape.append([0])
                                    else:
                                        bias_shape.append(layer.tresh.shape)    
                                else:
                                    bias_shape.append([None])
                            else:
                                weights_shape.append(layer.weight.shape)
                                if layer.bias!=None:
                                    bias_shape.append(layer.bias.shape)
                                else:
                                    bias_shape.append([None])           

            # unpack node
            else:
                subhandles, subbase, subweight,subbias = self._traverse_model_set_hooks(
                    layer, layer_types
                )
                for i in subhandles:
                    handles.append(i)
                for i in subbase:
                    output_shape.append(i)
                for i in subweight:
                    weights_shape.append(i)
                for i in subbias:
                    bias_shape.append(i)   

        return (handles, output_shape, weights_shape,bias_shape)

    @staticmethod
    def _fit_index(tensor, corrupt_idx):
        """把故障位置索引适配到张量的实际维度(防御性)。

        故障位置是按"补齐到 4 维"的 shape 生成的, 不足的维度填 None; 而
        Relu_bound 的 bounds/tresh 在 bounds_type='layer' 时是标量(0 维),
        'neuron' 时是向量(1 维), 直接用 4 元组索引会 IndexError。
        """
        dim = tensor.dim()
        if dim == 0:
            return ()
        pos = [i for i in corrupt_idx if isinstance(i, int)][:dim]
        while len(pos) < dim:
            pos.append(0)
        return tuple(pos)

    def _static_collect(self, model, layer_types):
        """按 children() 递归顺序纯静态收集匹配类型的叶子层(不依赖 hook)。"""
        layers = []
        for layer in model.children():
            if list(layer.children()) == []:
                for i in layer_types:
                    if isinstance(layer, i):
                        layers.append(layer)
                        break
            else:
                layers.extend(self._static_collect(layer, layer_types))
        return layers

    def _collect_inj_layers(self, model, layer_types):
        """收集真正参与注入的层, 顺序与 self.output_size / layers_type 严格一致。

        纯静态遍历会与 hook 记录错位, 原因有二:
          1) MultiheadAttention.out_proj 等层被 F.multi_head_attention_forward
             内部以 F.linear() 直接计算, 模块 forward 不执行 -> hook 不触发;
          2) children() 递归顺序与 forward 实际执行顺序本身也不同(例如本轨迹模型
             encoder.fuse 静态遍历排第 9, 但在 forward 中是第 2 个执行的)。
        因此这里按 profiling 记录的触发位置 _hooked_pos 从静态列表中取层,
        保证层编号与 output_size 对齐, 避免把故障注入到错误的层。
        """
        all_layers = self._static_collect(model, layer_types)
        if getattr(self, "_hooked_pos", None) is None:
            return all_layers
        return [all_layers[i] for i in self._hooked_pos if i < len(all_layers)]

#该函数的作用是遍历整个模型的每一层，在特定层上注册钩子，这些钩子会在前向传播时被调用，从而实现
    #对网络中特定层的故障注入
    def _traverse_model_set_hooks_neurons(self, model, layer_types, customInj, injFunc):
        handles = [] #用于存储层钩子列表
        for layer in model.children(): #遍历模型的所有子模块（子层）
            # leaf node
            #判断该层是否是叶子结点（即没有子层）。叶子节点是指没有子模块的层
            # （例如卷积层，线性层）
            if list(layer.children()) == []:
                #如果layer_types中包含“all”,表示对所有层进行故障注入
                if "all" in layer_types:
                    #如果使用了自定义注入函数（customInj=True），则使用injFunc,否则使用
                    #默认的_set_value函数作为钩子（这个函数没有看） 实现神经元级故障？
                    #每个层在执行完推理之后自定执行hook函数即进行故障注入？

                    hook = injFunc if customInj else self._set_value
                    handles.append(layer.register_forward_hook(hook))
                else:
                    #如果layer_types中指定了某些分层类型（例如卷积层，全连接层），
                    #那么只有这些层会被注入故障
                    for i in layer_types:
                        if isinstance(layer, i):
                            hook = injFunc if customInj else self._set_value
                            handles.append(layer.register_forward_hook(hook))
            # unpack node
            #如果当前层是一个非叶子节点，（例如包含子层的层），递归遍历子层，继续注册钩子
            else:
                subHandles = self._traverse_model_set_hooks_neurons(
                    layer, layer_types, customInj, injFunc
                )
                for i in subHandles:
                    handles.append(i)
        #返回的handles列表的具体内容
        #handls列表中包含了所有在模型中注册的钩子句柄，每个钩子句柄对应一个层，在
        #该层前向传播的时候会执行注册的钩子函数
        return handles


    def declare_weight_fault_injection(self, **kwargs):
        self._reset_fault_injection_state()
        custom_injection = False
        custom_function = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, custom_function = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", [])
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_bit = kwargs.get("dim4",[])
                total_bits = kwargs.get("total_bits")
                n_frac = kwargs.get("n_frac")
                n_int = kwargs.get("n_int")
            else:
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_value = kwargs.get("value", [])
                corrupt_bit = kwargs.get("dim4",[])
                total_bits = kwargs.get("total_bits")
                n_frac = kwargs.get("n_frac")
                n_int = kwargs.get("n_int")
        else:
            raise ValueError("Please specify an injection or injection function")

        # TODO: bound check here

        self.corrupted_model = copy.deepcopy(self.original_model)

        current_weight_layer = 0
        for layer in self._collect_inj_layers(self.corrupted_model, self._inj_layer_types):
            if isinstance(layer, tuple(self._inj_layer_types)):
                inj_list_weight = list(
                    filter(
                        lambda x: corrupt_layer[0][x] == current_weight_layer,
                        range(len(corrupt_layer[0])),
                    )
                )
                inj_list_bias = list(
                    filter(
                        lambda x: corrupt_layer[1][x] == current_weight_layer,
                        range(len(corrupt_layer[1])),
                    )
                )
                for inj in inj_list_weight:

                    corrupt_idx = tuple(
                        [
                            corrupt_k[0][inj],
                            corrupt_c[0][inj],
                            corrupt_kH[0][inj],
                            corrupt_kW[0][inj],
                        ]
                    )
                    # print(corrupt_idx)
                    corrupt_bit_idx = corrupt_bit[0][inj]
                    if isinstance(layer,Relu_bound):
                            _bi = self._fit_index(layer.bounds, corrupt_idx)
                            orig_value = layer.bounds[_bi].item()
                            with torch.no_grad():
                                if custom_injection:
                                    corrupt_value = custom_function[1](layer.bounds, _bi,corrupt_bit_idx ,total_bits ,n_frac , n_int)
                                    layer.bounds[_bi].copy_(corrupt_value)
                                else:
                                    layer.bounds[_bi].copy_(corrupt_value[inj]) 
                    else:
                        orig_value = layer.weight[corrupt_idx].item()   
                        with torch.no_grad():
                            if custom_injection:
                                corrupt_value = custom_function[0](layer.weight, corrupt_idx,corrupt_bit_idx,total_bits ,n_frac , n_int)
                                # print(corrupt_value)
                                layer.weight[corrupt_idx].copy_(corrupt_value)
                            else:
                                layer.weight[corrupt_idx].copy_(corrupt_value[inj])

                for inj in inj_list_bias:
                    corrupt_idx = tuple(
                        [
                            corrupt_k[1][inj],
                            corrupt_c[1][inj],
                            corrupt_kH[1][inj],
                            corrupt_kW[1][inj],
                        ]
                    )
                    corrupt_bit_idx = corrupt_bit[1][inj]
                    # print(corrupt_idx)
                    if  isinstance(layer,Relu_bound):
                        _ti = self._fit_index(layer.tresh, corrupt_idx)
                        orig_value = layer.tresh[_ti].item()
                        with torch.no_grad():
                            if custom_injection:
                                corrupt_value = custom_function[1](layer.tresh, _ti ,corrupt_bit_idx ,total_bits,n_frac , n_int)
                                layer.tresh[_ti].copy_(corrupt_value)
                            else:
                                layer.tresh[_ti].copy_(corrupt_value[inj])
                    else:    
                        orig_value = layer.bias[corrupt_idx].item()   
                        with torch.no_grad():
                            if custom_injection:
                                corrupt_value = custom_function[1](layer.bias, corrupt_idx ,corrupt_bit_idx , total_bits,n_frac , n_int)
                                layer.bias[corrupt_idx].copy_(corrupt_value)
                            else:
                                layer.bias[corrupt_idx].copy_(corrupt_value[inj])

                current_weight_layer += 1
        return self.corrupted_model

#这段代码实现了神经网络中“故障注入”（Fault injection）的过程。
#这段代码用于在训练过程中人为地在模型的不同层，不同批次以及不同维度引入故障，
#可能包括设定某些值为无效值。（比如设为Nan 或 inf）

# 定义了一个方法delcare_neuron_fault,该方法接受任意数量的关键字参数（**kwargs）用于指定故障注入
    #的具体参数
    def declare_neuron_fault_injection(self, **kwargs):
        #调用self._rest_faul_injection_state()来重置故障注入状态，可能是将之前的故障注入清除
        #确保注入状态的干净
        self._reset_fault_injection_state()
        #是否使用了自定义的故障注入函数
        custom_injection = False #是否启动了自定义的故障注入函数
        injection_function = False #故障注入函数对象

        #是否传入了关键字参数kwargs
        if kwargs:
            #如果kwargs中包含function参数，表示用户传入了自定义的故障注入函数，

            if "function" in kwargs:
                #打印日志“Declaring Custom Function”
                logging.info("Declaring Custom Function")
                custom_injection, injection_function = True, kwargs.get("function")
            else:
                #日志打印 说明是使用预定的故障注入方式
                logging.info("Declaring Specified Fault Injector")
                self.corrupt_value = kwargs.get("value", [])
                #从kwargs中，获取value参数，指定故障的具体指，默认为空列表

            #故障注入的分层
            self.corrupt_layer = kwargs.get("layer_num", [])
            #故障注入的批次
            self.corrupt_batch = kwargs.get("batch", [])
            #故障注入的维度 例如：卷积层的深度，高度，宽度
            self.corrupt_dim[0] = kwargs.get("dim1", [])
            self.corrupt_dim[1] = kwargs.get("dim2", [])
            self.corrupt_dim[2] = kwargs.get("dim3", [])

            #打印日志显示，故障注入的层，批次和维度信息便于调试
            logging.info(f"Convolution: {self.corrupt_layer}")
            logging.info("Batch, x, y, z:")
            logging.info(
                f"{self.corrupt_batch}, {self.corrupt_dim[0]}, {self.corrupt_dim[1]}, {self.corrupt_dim[2]}"
            )
        else:
            #如果kwargs为空，抛出一个错误，提示用户必须指定故障注入的内容或注入函数
            raise ValueError("Please specify an injection or injection function")

        #检查故障注入的合法性，确保指定的批次，层次和维度是有效的，不会超出网络的实际范围
        self.check_bounds(
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
        )

        #深拷贝一个故障注入的模型 确保故障注入后不会对原有模型产生音响
        self.corrupted_model = copy.deepcopy(self.original_model)

        #调用_traverse_mdel_set_hooks_neurons方法，在模型中遍历并为每个神经元设置钩子，
        #钩子通常用于在前向传播过程中插入定义的操作，如故障注入
        handles_neurons = self._traverse_model_set_hooks_neurons(
            self.corrupted_model,#要进行故障注入的模型
            self._inj_layer_types, #故障注入的层类型
            custom_injection, #是否使用了自定义故障注入函数
            injection_function, #自定义的故障注入函数对象（如果有）
        )

        #将为每个神经元设置的钩子存储到self.handeles，以便后续的管理和清除
        for i in handles_neurons:
            self.handles.append(i)

        return self.corrupted_model

    def check_bounds(self, batch, layer, dim):
        if (
            len(batch) != len(layer)
            or len(batch) != len(dim[0])
            or len(batch) != len(dim[1])
            or len(batch) != len(dim[2])
        ):
            raise AssertionError("Injection location missing values.")

        logging.info("Checking bounds before runtime")
        for i in range(len(batch)):
            self.assert_injection_bounds(i)

#功能：
#边界检查：该函数通过检查注入的索引是否在合法范围内，确保故障注入操作不会越界，避免在运行过程中产生
    #不可预期的错误
    #支持多维度层：根据层的维度（2d,3d,4d），它检查每个维度的索引是否超出该层的实际维度范围
    #警告信息：对于低纬度的层，它会忽略无效的维度索引并发出警告
    #日志记录：函数会记录故障注入索引的验证过程，帮助调试和跟踪问题
    def assert_injection_bounds(self, index: int):
        if index < 0:
            raise AssertionError(f"Invalid injection index: {index}")
        if self.corrupt_batch[index] >= self.batch_size:
            raise AssertionError(
                f"{self.corrupt_batch[index]} < {self.batch_size()}: Invalid batch element!"
            )
        if self.corrupt_layer[index] >= len(self.output_size):
            raise AssertionError(
                f"{self.corrupt_layer[index]} < {len(self.output_size)}: Invalid layer!"
            )

        corrupt_layer_num = self.corrupt_layer[index]
        layer_type = self.layers_type[corrupt_layer_num]
        layer_dim = self.layers_dim[corrupt_layer_num]
        layer_shape = self.output_size[corrupt_layer_num]

        for d in range(1, 4):
            if layer_dim > d and self.corrupt_dim[d - 1][index] >= layer_shape[d]:
                raise AssertionError(
                    f"{self.corrupt_dim[d - 1][index]} < {layer_shape[d]}: Out of bounds error in Dimension {d}!"
                )

        if layer_dim <= 2 and (
            self.corrupt_dim[1][index] is not None
            or self.corrupt_dim[2][index] is not None
        ):
            warnings.warn(
                f"Values in Dim2 and Dim3 ignored, since layer is {layer_type}"
            )

        if layer_dim <= 3 and self.corrupt_dim[2][index] is not None:
            warnings.warn(f"Values Dim3 ignored, since layer is {layer_type}")

        logging.info(f"Finished checking bounds on inj '{index}'")

#这段代码定义了一个_set_value函数，它是一个钩子函数，在模型的前向传播过程中，主要的作用是
#注入故障（即修改某一层的激活值）。故障注入的具体细节取决于当前层的维度（2d,3d,4d），
    #以及注入的具体位置。
    def _set_value(self, module, input_val, output):
        #model 当前正在处理的层（模块）,该层的输入数据，该层的输出数据


        #这一行记录了日志，打印当前正在处理的层编号和该层的类型
        logging.info(
            f"Processing hook of Layer {self.current_layer}: {self.layers_type[self.current_layer]}"
        )
        #inj_list是一个索引列表，包含了所有需要进行故障注入的神经元/元素在当前层的位置
        #corrupt_layer记录了哪些层需要故障注入，self.current_layer表示当前处理的层
        #这里通过filter过滤出当前层要注入故障的神经元索引
        inj_list = list(
            filter(
                lambda x: self.corrupt_layer[x] == self.current_layer,
                range(len(self.corrupt_layer)),
            )
        )

        #获取当前层的维度信息（可能是2d，3d或4d）
        #self.layers_dim是一个包含每层维度信息的列表
        layer_dim = self.layers_dim[self.current_layer]
        #记录日志，打印当前层需要注入故障的元素
        logging.info(f"Layer {self.current_layer} injection list size: {len(inj_list)}")
        #针对不同维度的注入
        #这段代码接下来的逻辑根据层的维度（layer_dim）来判断故障注入的方式，并且遍历注入故障的元素
        #修改它们的值。代码分为3个不同的维度情况：2d,3d或4d


        #对于2d层：
        #if layer_dim=2:判断当前层的维度是2d，即二位矩阵
        #inj_list中的每个元素对应需要故障注入的位置
        if layer_dim == 2:
            for i in inj_list:
                #在修改输出值之前，先检查注入位置是否有效。
                #这个方法应该是用来验证注入的索引是否在有效范围内
                #  （具体实现可能是一个边界检查函数）
                self.assert_injection_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]]}"
                )
                #日志记录原始值，并将该位置的输出值修改为self.corrupt_value[i]即故障注入值
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                #使用注入位置的索引来修改输出张量的特定值，这里self.corrupt_batch[i]是
                # 批次索引 ，self.corrupt_dim[0][i]是特定神经元索引
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 3:
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]]}"
                )
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i], self.corrupt_dim[1][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 4:
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}"
                )
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = self.corrupt_value[i]

        self.update_layer()

    def _save_output_size(self, module, input_val, output):
        shape = list(output.size())
        dim = len(shape)

        self.layers_type.append(type(module))
        self.layers_dim.append(dim)
        self.output_size.append(shape)
        if hasattr(self, "_hooked_modules"):
            self._hooked_modules.append(module)

    def update_layer(self, value=1):
        self.current_layer += value

    def reset_current_layer(self):
        self.current_layer = 0

    def get_weights_size(self, layer_num):
        return self.weights_size[layer_num]
    def get_bias_size(self,layer_num):
        return self.bias_size[layer_num]
    def get_weights_dim(self, layer_num):
        return len(self.weights_size[layer_num])

    def get_layer_type(self, layer_num):
        return self.layers_type[layer_num]

    def get_layer_dim(self, layer_num):
        return self.layers_dim[layer_num]

    def get_layer_shape(self, layer_num):
        return self.output_size[layer_num]
    def get_total_bits(self):
        return self.total_bits , self.n_frac,self.n_int
    def get_total_layers(self):
        return len(self.output_size)

    def get_tensor_dim(self, layer, dim):
        if dim > len(self.layers_dim):
            raise AssertionError(f"Dimension {dim} is out of bounds for layer {layer}")
        return self.output_size[layer][dim]

    def print_pytorchfi_layer_summary(self):
        summary_str = (
            "============================ PYTORCHFI INIT SUMMARY =============================="
            + "\n\n"
        )

        summary_str += "Layer types allowing injections:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for l_type in self._inj_layer_types:
            summary_str += "{:>5}".format("- ")
            substring = str(l_type).split(".")[-1].split("'")[0]
            summary_str += substring + "\n"
        summary_str += "\n"

        summary_str += "Model Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )

        summary_str += "   - Shape of input into the model: ("
        for dim in self._input_shape:
            summary_str += str(dim) + " "
        summary_str += ")\n"

        summary_str += "   - Batch Size: " + str(self.batch_size) + "\n"
        summary_str += "   - CUDA Enabled: " + str(self.use_cuda) + "\n\n"

        summary_str += "Layer Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        line_new = "{:>5}  {:>35}  {:>10} {:>20} {:>20} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Weight Shape", "Output Shape","Bias Shape"
        )
        summary_str += line_new + "\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for layer, _dim in enumerate(self.output_size):
            line_new = "{:>5}  {:>35}  {:>10} {:>20} {:>20} {:>20}".format(
                layer,
                str(self.layers_type[layer]).split(".")[-1].split("'")[0],
                str(self.layers_dim[layer]),
                str(list(self.weights_size[layer])),
                str(self.output_size[layer]),
                str(list(self.bias_size[layer]))
            )
            summary_str += line_new + "\n"

        summary_str += (
            "=================================================================================="
            + "\n"
        )

        logging.info(summary_str)
        return summary_str

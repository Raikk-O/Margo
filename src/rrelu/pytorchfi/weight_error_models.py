

import random
from fxpmath import Fxp
import rrelu.pytorchfi.core as core 
from rrelu.pytorchfi.util import random_value
import logging
import torch
import numpy as np


def random_weight_location(pfi, layer: int = -1):

    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_weights_dim(layer) 
    shape = pfi.get_weights_size(layer) 

    dim0_shape = shape[0]
    k = random.randint(0, dim0_shape - 1)
    if dim > 1:
        dim1_shape = shape[1]
        dim1_rand = random.randint(0, dim1_shape - 1)
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)
    else:
        dim2_rand = None
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)
    else:
        dim3_rand = None

    return ([layer], [k], [dim1_rand], [dim2_rand], [dim3_rand])


def random_weight_inj(
    pfi, corrupt_layer: int = -1, min_val: int = -1, max_val: int = 1
):
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_layer)
    faulty_val = [random_value(min_val=min_val, max_val=max_val)]

    return pfi.declare_weight_fault_injection(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )



def zero_func_rand_weight(pfi: core.FaultInjection):
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fault_injection(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def bit2float(b, device=torch.device('cpu'), num_e_bits=8, num_m_bits=23, bias=127.):
    expected_last_dim = num_m_bits + num_e_bits + 1
    dtype = torch.float32

    s = torch.index_select(b, -1, torch.arange(0, 1, device=device))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits, device=device))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits, device=device))
    
    out = (torch.pow((torch.ones(1, device=device) * -1), s)).squeeze(-1).type(dtype)
    
    exponents = -torch.arange(-(num_e_bits - 1.), 1., device=device)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * torch.pow(torch.ones(1, device=device) * 2, exponents), dim=-1) - bias
    out *= torch.pow(torch.ones(1, device=device) * 2, e_decimal)
    
    matissa = (torch.pow((torch.ones(1, device=device) * 2), -torch.arange(1., num_m_bits + 1., device=device))).repeat(m.shape[:-1] + (1,))
    
    out *= 1. + torch.sum(m * matissa, dim=-1)
    
    out[torch.abs(out) < 5e-30] = 0
    out[out == float("inf")] = 10e30
    out[out == float("-inf")] = -10e30
    out = out.detach()
    
    return out



def float2bit(f, device=torch.device('cpu'), num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    s = torch.sign(f)
    f = f * s
    
    s = (s * (-1) + 1.) * 0.5
    s = s.unsqueeze(-1)
    s[s == 0.5] = 0

    e_scientific = torch.floor(torch.log2(f))
    e_decimal = e_scientific + bias
    e = integer2bit(e_decimal, num_bits=num_e_bits, device=device)
    e = torch.nan_to_num(e)
    e[e == float("inf")] = 0
    e[e == float("-inf")] = 0
    int_precision = 1024
    m1 = integer2bit(f - f % 1, num_bits=int_precision, device=device)
    m2 = remainder2bit(f % 1, num_bits=bias, device=device)
    
    m = torch.cat([m1, m2], dim=-1)
    dtype = f.type()
    
    idx = torch.arange(num_m_bits, device=device).unsqueeze(0).type(dtype) + (float(int_precision) - e_scientific).unsqueeze(-1)

    idx[idx == float("inf")] = 0
    idx[idx == float("-inf")] = 0
    idx = torch.nan_to_num(idx)
    
    idx = idx.long()
    
    m = torch.gather(m, dim=-1, index=idx)
    
    out = torch.cat([s, e, m], dim=-1).type(dtype)
    out = out.detach()
    return out


def remainder2bit(remainder, num_bits=127, device=torch.device('cpu')):
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits, device=device).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer, num_bits=32, device=torch.device('cpu')):
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1, device=device).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2


def quantized_int2bit(qint_tensor, qbit, device=torch.device('cpu')):       
    qint_tensor = qint_tensor.type(torch.int)
    mask = 2 ** torch.arange(qbit - 1, -1, -1).to(qint_tensor.device, qint_tensor.dtype)
    return qint_tensor.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.float)


def quantized_bit2int(qbinary_tensor, qbit, device=torch.device('cpu')):    
    mask = 2 ** torch.arange(qbit - 1, -1, -1).to(qbinary_tensor.device, qbinary_tensor.dtype)
    mask[0] = -mask[0]      
    return torch.sum(mask * qbinary_tensor, -1)

def _zero_rand_weight(data, location):
    new_data = data[location] * 0
    return new_data

def twos_comp(val, bits):
    if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
    return val

def twos_comp_shifted(val, nbits):
    return (1 << nbits) + val if val < 0 else twos_comp(val, nbits)


def flip_bit_signed(orig_value, max_value, bit_pos):
        save_type = orig_value.dtype
        total_bits = 31
        logging.info(f"Original Value: {orig_value}")

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = twos_comp_shifted(quantum, total_bits)  
        logging.info(f"Quantum: {quantum}")
        logging.info(f"Twos Couple: {twos_comple}")

        bits = bin(twos_comple)[2:]
        logging.info(f"Bits: {bits}")

        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info(f"Sign extend bits {bits}")

        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info(f"New bits: {bits_str_new}")

        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = twos_comp(new_quantum, total_bits)
        logging.info(f"Out: {out}")

        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info(f"New Value: {new_value}")

        return torch.tensor(new_value, dtype=save_type)

    
 


def bit_flip_weight_fixed(data, location , bit = None , total_bits = None ,n_frac = None , n_int = None):
    orig_value = data[location].item() 
    
    total_bits = total_bits   
    x = Fxp(orig_value, True, n_word=total_bits,n_frac=n_frac,n_int=n_int)
    bits = x.bin()
     
    bits_new = list(bits)
    
    bit_pos = bit   
    
    bit_loc = total_bits - bit_pos - 1
    if bits_new[bit_loc] == '0':
        bits_new[bit_loc] = '1'
    else:
        bits_new[bit_loc] = '0'
    bits_str_new = "".join(bits_new)
    if not bits_str_new.isdigit():
        print("Error: Not all the bits are digits (0/1)")

    assert bits_str_new.isdigit()
    FXP_value = x("0b"+bits_str_new)
    new_value = FXP_value.astype(float)

    return torch.tensor(new_value)
def int2bin(input, num_bits):
    
    output = input.clone()
    if num_bits == 1: 
        output = output/2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2**num_bits + output[input.lt(0)]

    return output


def bin2int(input, num_bits):
    
    if num_bits == 1:
        output = input*2-1
    elif num_bits > 1:
        mask = 2**(num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    return output


def bit_flip_weight_int(data, location , bit = None , total_bits = None ,n_frac = None , n_int = None):
    orig_value = torch.tensor([data[location].item()],dtype=torch.int) 
    twos_comple = twos_comp_shifted(orig_value, total_bits)  
    bits = bin(twos_comple)[2:]
    temp = "0" * (total_bits - len(bits))
    bits = temp + bits
    if len(bits) != total_bits:
        raise AssertionError
        
    if bit >= total_bits:
        raise AssertionError

    bits_new = list(bits)
    bit_loc = total_bits - bit - 1
    if bits_new[bit_loc] == "0":
        bits_new[bit_loc] = "1"
    else:
        bits_new[bit_loc] = "0"
    bits_str_new = "".join(bits_new)
    

    if not bits_str_new.isdigit():
        logging.info("Error: Not all the bits are digits (0/1)")

    if not bits_str_new.isdigit():
        raise AssertionError
    new_quantum = int(bits_str_new, 2)
    out = twos_comp(new_quantum, total_bits)
    return torch.tensor(out)

def bit_flip_weight_IEEE(data, location , bit = None , total_bits = None ,n_frac = None , n_int = None):
    orig_value = torch.tensor([data[location].item()]) 
    total_bits = total_bits    
    bit_pos = bit
    bits= float2bit(orig_value)
    bit_loc = total_bits - bit_pos - 1
    bits[0,bit_loc] = torch.logical_xor(bits[0,bit_loc],torch.tensor(1.0)).float()    
    new_value = bit2float(bits)

    return torch.tensor(new_value[0])


def multi_weight_inj_int(pfi, sdc_p=1e-5, function1=bit_flip_weight_int,function2=bit_flip_weight_IEEE):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        shape_bias = list(pfi.get_bias_size(layer_idx))
        dim_bias = len(shape_bias)
        dim_len = len(shape)  
        shape.extend([1 for i in range(4 - len(shape))])
        nunmber_fault_weight = int(shape[0] * shape[1] * shape[2] * shape [3] * total_bits * sdc_p)
        shape_bias.extend([1 for i in range(4 - len(shape_bias))])
        if shape_bias[0] !=None : 
           nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * shape_bias[2] * shape_bias [3] * total_bits * sdc_p) 
        if nunmber_fault_weight !=0: 
            k_w = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
            dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
            dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
            dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
            dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
        if shape_bias[0]!=None:
            if nunmber_fault_bias!=0:
                k_b = torch.randint(shape[0],(nunmber_fault_bias,), device='cuda')
                dim1_b = torch.randint(shape[1],(nunmber_fault_bias,), device='cuda')
                dim2_b = torch.randint(shape[2],(nunmber_fault_bias,), device='cuda')
                dim3_b = torch.randint(shape[3],(nunmber_fault_bias,), device='cuda')
                dim4_b = torch.randint(total_bits,(nunmber_fault_bias,), device='cuda')
        for fault in range(nunmber_fault_weight):
            idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
            for i in range(dim_len + 1):
                corrupt_idx[i].append(idx[i])
            for i in range(dim_len + 1, 5): 
                corrupt_idx[i].append(None)  
            corrupt_bit_idx.append(dim4_w[fault]) 
        if shape_bias[0]!= None:                                                                       
            for fault in range(nunmber_fault_bias):
                idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                for i in range(dim_bias + 1):
                    corrupt_idx_bias[i].append(idx[i])
                for i in range(dim_bias + 1, 5): 
                    corrupt_idx_bias[i].append(None)  
                corrupt_bit_idx_bias.append(dim4_b[fault])       
                                                                     
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = None,
        n_int = None
    )



def multi_weight_inj_float(pfi, sdc_p=1e-5, function1=bit_flip_weight_IEEE,function2=bit_flip_weight_IEEE):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        shape_bias = list(pfi.get_bias_size(layer_idx))
        dim_bias = len(shape_bias)
        dim_len = len(shape)  
        shape.extend([1 for i in range(4 - len(shape))])
        nunmber_fault_weight = int(shape[0] * shape[1] * total_bits * sdc_p)
        shape_bias.extend([1 for i in range(4 - len(shape_bias))])
        if shape_bias[0] !=None : 
           nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * total_bits * sdc_p)
        if nunmber_fault_weight !=0:   
            k_w = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
            dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
            dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
            dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
            dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
        if shape_bias[0]!=None:
            if nunmber_fault_bias!=0:
                k_b = torch.randint(shape[0],(nunmber_fault_bias,), device='cuda')
                dim1_b = torch.randint(shape[1],(nunmber_fault_bias,), device='cuda')
                dim2_b = torch.randint(shape[2],(nunmber_fault_bias,), device='cuda')
                dim3_b = torch.randint(shape[3],(nunmber_fault_bias,), device='cuda')
                dim4_b = torch.randint(total_bits,(nunmber_fault_bias,), device='cuda')
        for fault in range(nunmber_fault_weight):
            idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
            for i in range(dim_len + 1):
                corrupt_idx[i].append(idx[i])
            for i in range(dim_len + 1, 5): 
                corrupt_idx[i].append(None)  
            corrupt_bit_idx.append(dim4_w[fault]) 
        if shape_bias[0]!= None:                                                                       
            for fault in range(nunmber_fault_bias):
                idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                for i in range(dim_bias + 1):
                    corrupt_idx_bias[i].append(idx[i])
                for i in range(dim_bias + 1, 5): 
                    corrupt_idx_bias[i].append(None)  
                corrupt_bit_idx_bias.append(dim4_b[fault])                                                           
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = None,
        n_int = None
    )



def multi_weight_inj_fixed(pfi, sdc_p=1e-5, function1=bit_flip_weight_fixed,function2=bit_flip_weight_fixed):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        shape_bias = list(pfi.get_bias_size(layer_idx))
        dim_bias = len(shape_bias)
        dim_len = len(shape)  
        shape.extend([1 for i in range(4 - len(shape))])
        nunmber_fault_weight = int(shape[0] * shape[1] * shape[2] * shape [3] * total_bits * sdc_p)
        shape_bias.extend([1 for i in range(4 - len(shape_bias))])
        if shape_bias[0] !=None : 
           nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * shape_bias[2] * shape_bias [3] * total_bits * sdc_p) 
        if nunmber_fault_weight !=0:      
            k_w = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
            dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
            dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
            dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
            dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
        if shape_bias[0]!=None:
            if nunmber_fault_bias!=0:
                k_b = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
                dim1_b = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
                dim2_b = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
                dim3_b = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
                dim4_b = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
        for fault in range(nunmber_fault_weight):
            idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
            for i in range(dim_len + 1):
                corrupt_idx[i].append(idx[i])
            for i in range(dim_len + 1, 5): 
                corrupt_idx[i].append(None)  
            corrupt_bit_idx.append(dim4_w[fault]) 
        if shape_bias[0]!= None:                                                                       
            for fault in range(nunmber_fault_bias):
                idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                for i in range(dim_bias + 1):
                    corrupt_idx_bias[i].append(idx[i])
                for i in range(dim_bias + 1, 5): 
                    corrupt_idx_bias[i].append(None)  
                corrupt_bit_idx_bias.append(dim4_b[fault])           
                                                                     
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = n_frac, 
        n_int = n_int, 
    )




def binaryOfFraction(fraction):
    binary = str()
    while (fraction):
        fraction *= 2
        if (fraction >= 1):
            int_part = 1
            fraction -= 1
        else:
            int_part = 0
        binary += str(int_part)
    return binary
def floatingPoint(real_no):
    sign_bit = 0
    if (real_no < 0):
        sign_bit = 1
    real_no = abs(real_no)
    int_str = bin(int(real_no))[2:]
    if real_no - int(real_no) !=0.0:
        fraction_str = binaryOfFraction(real_no - int(real_no))
    else:
        fraction_str=''    
    if int(real_no)==0:
        ind = int_str.index('0')
    else:
        ind = int_str.index('1')
    exp_str = bin((len(int_str) - ind - 1) + 127)[2:]
    mant_str = int_str[ind + 1:] + fraction_str
    mant_str = mant_str + ('0' * (23 - len(mant_str)))
    ieee_32 = str(sign_bit) + exp_str  + mant_str
    return ieee_32

if __name__=="__main__":
    a = torch.tensor([127])
    print(bit_flip_weight_int(a,0,8,8))

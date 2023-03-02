import math
import torch
import logging
import numpy as np
from pathlib import Path

module_logger = logging.getLogger(f'onnx_profiling.onnx_profile.onnx_counter.{str(Path(__file__).stem)}')

def counter_parameters(para_list):
    total_params = 0
    for p in para_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params

def counter_zero_ops():
    return torch.DoubleTensor([int(0)])

def counter_conv(bias, kernel_size, output_size, in_channel, group):
    """inputs are all numbers!"""
    #print("Bias, Kernel_size, Output_size, In_channel, Group : ", bias, kernel_size, output_size, in_channel, group)
    module_logger.debug("Conv Layer Params: Bias, Kernel_size, Output_size, In_channel, Group -> %s, %s, %s, %s, %s", bias, kernel_size, output_size, in_channel, group)
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size)])

def counter_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def counter_relu(input_size: torch.Tensor):
    return torch.DoubleTensor([int(input_size)])

def counter_nms(n_dimension: np.int64, scores: np.ndarray):
    ops_box_area = 3
    ops_union = 2 * ops_box_area + 2
    ops_inter = 7
    ops_box_iou = (ops_inter + ops_union + 1) * n_dimension
    total_ops = ops_box_iou + scores[1] + scores[2]
    return torch.DoubleTensor([int(total_ops)])

def counter_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (batch_size-1) * (total_exp + total_add + total_div)
    return torch.DoubleTensor([total_ops])

def counter_tanh(batch, nfeatures):
    exp_neg = nfeatures
    exp_pos = nfeatures
    numerator_sub = nfeatures
    denominator_add = nfeatures
    total_div = nfeatures
    total_ops = batch * (exp_neg + exp_pos + numerator_sub + denominator_add + total_div)
    return torch.DoubleTensor([total_ops])

def counter_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])

def counter_adap_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])

def counter_upsample(mode: str, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= ops_solve_A + ops_solve_p
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return torch.DoubleTensor([int(total_ops)])

def counter_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]

def counter_mul(input_size):
    return input_size

def counter_pow(input_size):
    return input_size

def counter_sqrt(input_size):
    return input_size

def counter_div(input_size):
    return input_size

def counter_misc(input_size):
    """input is a number not a array or tensor"""
    return input_size

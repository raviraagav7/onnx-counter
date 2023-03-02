from distutils.command.config import LANG_EXT
import sys
from pandas import array
import torch
import logging
import numpy as np
from pathlib import Path
from onnx import numpy_helper
from thop.utils import parse_attribute, pad_impl
from thop.vision.basic_hooks import zero_ops
from .counter import (
    counter_matmul,
    counter_zero_ops,
    counter_conv,
    counter_mul,
    counter_norm,
    counter_pow,
    counter_sqrt,
    counter_div,
    counter_softmax,
    counter_avgpool,
    counter_upsample,
    counter_misc,
    counter_nms,
    counter_tanh
)

module_logger = logging.getLogger(f'onnx_profiling.onnx_profile.{str(Path(__file__).stem)}')
WEIGHT_SIGMOID = 1
WEIGHT_TANH = 1

def onnx_counter_matmul(diction, node):
    """
    There are 𝑛×𝑚 elements in the output matrix. Each of them is obtained by 𝑝 multiplications
    (1 element from the first matrix and 1 from the second), then summing up. Since you have 𝑝 
    products, you add 𝑝−1 of them to the first one. So the number of operations for one element 
    in the output matrix is 𝑝 multiplications and 𝑝−1 additions, meaning 2𝑝−1 FLOPS. Then for 
    all elements, you get 𝑛×𝑚×(2𝑝−1) FLOPS.
    Ref: https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    """
    input1 = node.input[0]
    input2 = node.input[1]
    input1_dim = diction[input1]
    input2_dim = diction[input2]
    module_logger.debug("Matmul input shape: %s, %s", input1_dim, input2_dim)
    out_size = np.append(input1_dim[0:-1], input2_dim[-1])
    output_name = node.output[0]
    n = input1_dim[0]
    p = input1_dim[1]
    m = input2_dim[1]
    flops = counter_misc(n * m * ((2*p) -1))
    # flops = counter_matmul(input1_dim, out_size[-2:])
    return flops, out_size, output_name

def onnx_counter_slice(diction, node, param):
    module_logger.debug("Slice node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    dict_param = dict(param)
    starts = dict_param.get(node.input[1], 0)
    ends = dict_param.get(node.input[2], 9223372036854775807)
    axes = dict_param.get(node.input[3], 0)
    steps = dict_param.get(node.input[4], 0)
    out_size = input_size.copy()
    out_size[axes] = np.array(list(range(out_size[axes])))[starts:ends:steps].shape[0]
    flops = torch.DoubleTensor([input_size[axes]])
    module_logger.debug("Slice node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_gather(diction, node, param):
    attr_dict = parse_attribute(node)
    module_logger.debug("Gather node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    dict_param = dict(param)
    indices = np.random.randint(np.prod(dict_param.get(node.input[1], 0)), size=tuple(dict_param.get(node.input[1], 0))) if isinstance(dict_param.get(node.input[1], 0), (np.ndarray, np.generic)) else dict_param.get(node.input[1], 0)
    data = np.random.randn(*tuple([1])).astype(np.float32)[0] if len(input_size) == 0 else np.random.randn(*tuple(input_size)).astype(np.float32)
    out_size = np.array(np.take(data, indices=indices, axis=attr_dict.get('axis', 0)).shape)
    flops = counter_misc(np.prod([input_size[attr_dict.get('axis', 0)], max(np.prod(indices), 1)]))
    module_logger.debug("Gather node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_topk(diction, node, param):
    attr_dict = parse_attribute(node)
    module_logger.debug("TopK node input : %s", diction[node.input[0]])
    output_name = node.output
    input_size = diction[node.input[0]].copy()
    dict_param = dict(param)
    K = dict_param.get(node.input[1][0], [1])[0]
    data = torch.rand(*input_size)
    values, indices = torch.topk(data, K)
    out_size = np.array(values.size()), np.array(indices.size())
    flops = counter_zero_ops()
    module_logger.debug("TopK node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_add(diction, node):
    module_logger.debug("Add node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    out_size = input_size.copy()
    flops = counter_misc(np.prod(input_size))
    module_logger.debug("Add node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_misc(diction, node):
    module_logger.debug("Misc node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    out_size = input_size.copy()
    flops = counter_misc(np.prod(input_size))
    module_logger.debug("Misc node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_conv(diction, node):
    # Ref : https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf
    dim_bias = 0
    input_count = 0
    for i in node.input:
        input_count += 1
    if input_count == 3:
        dim_bias = 1
        dim_weight = diction[node.input[1]]
    else:
        dim_weight = diction[node.input[1]]
    group = 1
    dim_pad = None
    dim_dil = None
    dim_stride = None
    group = None
    auto_pad = b'NOTSET'
    for attr in node.attribute:
        if attr.name == "auto_pad":
            auto_pad = attr.s
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        if attr.name == "strides":
            dim_stride = attr.ints
        if attr.name == "pads":
            dim_pad = attr.ints
        if attr.name == "dilations":
            dim_dil = attr.ints
        if attr.name == "group":
            group = attr.i
    if dim_stride is None:
        dim_stride = [1, 1]
    
    if dim_dil is None:
        dim_dil = [1, 1]
    if group is None:
        group = 1
    if dim_pad is None:
        dim_pad = [0, 0, 0, 0]
    dim_input = diction[node.input[0]].copy()
    output_size = np.append(
        dim_input[0 : -np.array(dim_kernel).size - 1], dim_weight[0]
    )
    hw = np.array(dim_input[-np.array(dim_kernel).size :])
    for i in range(hw.size):
        if auto_pad == b'NOTSET':
            # hw[i] = np.round(
            #     (hw[i] + 2 * dim_pad[i] - dim_dil[i] * (dim_kernel[i] - 1) - 1)
            #     / dim_stride[i]
            #     + 1
            # )
            hw[i] = np.floor(
                (hw[i] + 2 * dim_pad[i] - dim_dil[i] * (dim_kernel[i] - 1) - 1)
                / dim_stride[i]
                + 1
            )
        elif auto_pad == b'SAME_UPPER' or auto_pad == b'SAME_LOWER':
            hw[i] = np.ceil(float(hw[i]) / float(dim_stride[1]))
        elif auto_pad == b'VALID':
            hw[i] = np.ceil(float(hw[i] - dim_kernel[i] + 1) / float(dim_stride[1]))
    output_size = np.append(output_size, hw)
    module_logger.debug("Kernel Size, Input Size, Output Size, Input Channel, Group : %s, %s, %s, %s, %s", np.prod(dim_kernel), dim_input, output_size, dim_weight[1], group)
    flops = counter_conv(
        dim_bias, np.prod(dim_kernel), np.prod(output_size), dim_weight[1], group
    )
    output_name = node.output[0]
    # if '140' in diction:
    #     print("conv",diction['140'],output_name)
    return flops, output_size, output_name

def onnx_counter_conv_transpose(diction, node):
    dim_bias = 0
    input_count = 0
    for i in node.input:
        input_count += 1
    if input_count == 3:
        dim_bias = 1
        dim_weight = diction[node.input[1]]
    else:
        dim_weight = diction[node.input[1]]
    
    group = 1
    pads = None # [0, 0, 0, 0]
    dilations = None # [1, 1]
    stride = None # [1, 1]
    auto_pad = b'NOTSET'
    output_padding = None # [0, 0]
    for attr in node.attribute:
        if attr.name == "auto_pad":
            auto_pad = attr.s
        if attr.name == "kernel_shape":
            kernel_shape = attr.ints  # kw,kh
        if attr.name == "strides":
            stride = attr.ints
        if attr.name == "pads":
            pads = attr.ints
        if attr.name == "dilations":
            dilations = attr.ints
        if attr.name == "group":
            group = attr.ints
        if attr.name == "output_padding":
            output_padding = attr.ints
            
    if pads is None:
        pads = [0] * len(kernel_shape) * 2
    
    if dilations is None:
        dilations = [1] * len(kernel_shape)
        
    if stride is None:
        stride = [1] * len(kernel_shape)
        
    if output_padding is None:
        output_padding = [0] * len(kernel_shape)
        
    input_size = diction[node.input[0]].copy()
    output_size = np.append(
        input_size[0 : -np.array(kernel_shape).size - 1], dim_weight[1]
    )
    hw = np.array(input_size[-np.array(kernel_shape).size :])
    """
    output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    
    If (auto_pads == SAME_UPPER): 
        pads[start_i] = total_padding[i]/2; 
        pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else: 
        pads[start_i] = total_padding[i] - (total_padding[i]/2); 
        pads[end_i] = (total_padding[i]/2).
    """
    for i in range(hw.size):
        hw[i] = stride[i] * (hw[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[i] - pads[i+2]
    output_size = np.append(output_size, hw)
    module_logger.debug("Kernel Size, Input Size, Output Size, Input Channel, Group : %s, %s, %s, %s, %s", np.prod(kernel_shape), input_size, output_size, dim_weight[1], group)
    
    batch_size = input_size[0]
    conv_shape = input_size[2:]
    flops = batch_size * np.prod(dim_weight) * np.prod(conv_shape)
    output_name = node.output[0]
    return flops, output_size, output_name

def onnx_counter_constant(diction, node):
    # print("constant",node)
    flops = counter_zero_ops()
    output_name = node.output[0]
    output_size = [1]
    # print(flops, output_size, output_name)
    return flops, output_size, output_name

def onnx_counter_mul(diction, node):
    input_1 = diction[node.input[0]].copy()
    input_2 = diction[node.input[1]].copy()
    
    input_1 = input_1.astype(np.int64)
    input_2 = input_2.astype(np.int64)
    
    x = np.random.randn(*tuple(input_1))
    y = np.random.randn(*tuple(input_2))
    x = x.astype(np.float32) if isinstance(x, np.ndarray) else x
    y = y.astype(np.float32) if isinstance(y, np.ndarray) else y
    z = x * y 
    output_size = list(z.shape)
    #print("Mul input 1 : ", diction[node.input[0]])
    #print("Mul input 1 size : ", np.array(diction[node.input[0]]).size)
    #print("Mul input 2 : ", np.array(diction[node.input[1]]).size)
    #print("Mul input 2 size : ", diction[node.input[1]])
    if np.array(diction[node.input[1]]).size < np.array(diction[node.input[0]]).size:
        flops = counter_mul(np.prod(input_2))
    else:
        flops = counter_mul(np.prod(input_1))
    #output_size = diction[node.input[0]]
    output_name = node.output[0]
    return flops, output_size, output_name

def onnx_counter_bn(diction, node):
    # (x-x')/σ one sub op and one div op
        # and the shift γ and β
    input_size = diction[node.input[0]].copy()
    # flops = counter_norm(np.prod(input_size))
    flops = counter_misc(np.prod(input_size) / input_size[1] * 4)
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_relu(diction, node):
    #input_size = diction[node.input[0]]
    #flops = counter_zero_ops()
    input_size = diction[node.input[0]].copy()
    flops = counter_misc(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    module_logger.debug("Relu input_size : %s", input_size)
    module_logger.debug("Relu output_size: %s", output_size)
    # print(flops, output_size, output_name)
    # if '140' in diction:
    #     print("relu",diction['140'],output_name)
    return flops, output_size, output_name

def onnx_counter_reducemean(diction, node):
    keep_dim = 0
    for attr in node.attribute:
        if "axes" in attr.name:
            dim_axis = np.array(attr.ints)
        elif "keepdims" in attr.name:
            keep_dim = attr.i
    input_size = diction[node.input[0]].copy()
    flops = counter_zero_ops()
    output_name = node.output[0]
    if keep_dim == 1:
        output_size = input_size
    else:
        output_size = np.delete(input_size, dim_axis)
    # output_size = input_size
    return flops, output_size, output_name

def onnx_counter_sub(diction, node):
    input_size = diction[node.input[0]].copy()
    flops = counter_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_pow(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]].copy()
    else:
        input_size = diction[node.input[0]].copy()
    flops = counter_pow(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_sqrt(diction, node):
    input_size = diction[node.input[0]].copy()
    flops = counter_sqrt(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_div(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]].copy()
    else:
        input_size = diction[node.input[0]].copy()
    flops = counter_div(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_instance(diction, node):
    input_size = diction[node.input[0]].copy()
    flops = counter_norm(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_softmax(diction, node):
    input_size = diction[node.input[0]].copy()
    if (node.attribute):
        dim = node.attribute[0].i
    else:
        dim = 1
    nfeatures = input_size[dim]
    batch_size = np.prod(input_size) / nfeatures
    flops = counter_softmax(nfeatures, batch_size)
    output_name = node.output[0]
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_pad(diction, node):
    # # TODO add constant name and output real vector
    if len(node.input) == 1:
        input_size = diction[node.input[0]]
    else:
        if (np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size):
            input_size = diction[node.input[1]]
        else:
            input_size = diction[node.input[0]]
            
    pads = [0] * (2 * len(input_size))
    value = 0.0
    mode = 'constant'
    axes = None
    for attr in node.attribute:
        if attr.name == "mode":
            mode = attr.s
        if attr.name == "pads":
            pads = attr.ints
        if attr.name == "value":
            value = attr.f
        if attr.name == "axes":
            axes = attr.ints
    inp_data = np.random.randn(*tuple(input_size.astype(np.int64)))
    y = pad_impl(inp_data, np.array(pads), mode= mode.decode("utf-8") if isinstance(mode, bytes) else mode, constant_values=value, axes=axes)
    
            
    flops = counter_zero_ops()
    output_name = node.output[0]
    
    output_size = np.array(y.shape)
    # input_size = diction[node.input[0]].copy()
    # flops = counter_misc(np.prod(input_size))
    # output_name = node.output[0]
    # output_size = input_size
    return flops, output_size, output_name

def onnx_counter_averagepool(diction, node):
    # TODO add support of ceil_mode and floor
    flops = counter_avgpool(np.prod(diction[node.input[0]]))
    module_logger.debug("avgpool input : %s", diction[node.input[0]])
    output_name = node.output[0]
    dim_pad = None
    for attr in node.attribute:
        # print(attr)
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        elif attr.name == "strides":
            dim_stride = attr.ints
        elif attr.name == "pads":
            dim_pad = attr.ints
        elif attr.name == "dilations":
            dim_dil = attr.ints
            # print(dim_dil)
    dim_input = diction[node.input[0]].copy()
    hw = dim_input[-np.array(dim_kernel).size :]
    if dim_pad is not None:
        for i in range(hw.size):
            hw[i] = int((hw[i] + 2 * dim_pad[i] - dim_kernel[i]) / dim_stride[i] + 0.5)+ 1
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    else:
        for i in range(hw.size):
            hw[i] = int((hw[i] - dim_kernel[i]) / dim_stride[i] + 0.5) + 1
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    # print(flops, output_size, output_name)
    module_logger.debug("avgpool output : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_flatten(diction, node):
    # print(node)
    #flops = counter_zero_ops()
    output_name = node.output[0]
    axis = node.attribute[0].i
    input_size = diction[node.input[0]].copy()
    flops = counter_misc(np.prod(input_size))
    output_size = np.append(input_size[axis - 1], np.prod(input_size[axis:]))
    # print("flatten",output_size)
    return flops, output_size, output_name

def onnx_counter_reshape(diction, node, reshape):
    # print(node)
    #flops = counter_zero_ops()
    #perms = node.attribute[0].ints
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    module_logger.debug("Reshape Input : %s", input_size)
    output_size = np.array(reshape)
    # Case 1 : Reshape has parameter of another output ()
    if not reshape :
        #output_size = np.append(input_size[0], np.prod(input_size[1:]))
        output_size = diction[node.input[1]]
    # Case 2 : Reshape has constant parameter (parameter in constant node or initializer (weight))
    else:
        for i in range(len(reshape)):
            if reshape[i] == 0 :
                output_size[i] = input_size[i]
            elif reshape[i] == -1:
                if len(input_size) > len(reshape):
                    output_size[i] = np.prod(input_size[i:])
                elif i == 0:
                    output_size[i] = np.prod(input_size[:i])
                else:
                    output_size[i] = input_size[reshape[i]]
            else:
            ## Ravi - Fixed
                output_size[i] = reshape[i]
    flops = counter_misc(np.prod(input_size))
    module_logger.debug("Reshape Output : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_gemm(diction, node):
    # print(node)
    # Compute Y = alpha * A' * B' + beta * C
    # Ref : https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf
    input_size = diction[node.input[0]].copy()
    dim_weight = diction[node.input[1]].copy()
    
    flops = (np.prod(input_size) * dim_weight[0] )+ dim_weight[0]
    output_size = np.append(input_size[0:-1], dim_weight[1])
    #output_size = input_size
    module_logger.debug("Input Size, Dim Weight, Output_Size: %s, %s, %s", input_size, dim_weight, output_size)
    output_name = node.output[0]
    return flops, output_size, output_name

def onnx_counter_maxpool(diction, node):
    # TODO add support of ceil_mode and floor
    # print(node)
    input_size = 1
    input_size = diction[node.input[0]].copy()
    module_logger.debug("Maxpool input size : %s", input_size)
    #flops = counter_zero_ops()
    output_name = node.output[0]
    dim_pad = [0, 0, 0, 0]
    dim_dil = [1, 1]
    ceil_mode = 0
    dim_stride = [1, 1]
    for attr in node.attribute:
        # print(attr)
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        elif attr.name == "strides":
            dim_stride = attr.ints
        elif attr.name == "pads":
            dim_pad = attr.ints
        elif attr.name == "dilations":
            dim_dil = attr.ints
        elif attr.name == "ceil_mode":
            ceil_mode = attr.ints
            # print(dim_dil)
    dim_input = diction[node.input[0]].copy()
    hw = dim_input[-np.array(dim_kernel).size :]
    module_logger.debug("Maxpool hw : %s", hw)
    
    for i in range(hw.size):
        if ceil_mode == 1:
            hw[i] = np.ceil(((hw[i] + (2 * dim_pad[i]) - dim_dil[i] * (dim_kernel[i] - 1) -1) / dim_stride[i]) + 1)
        else:
            hw[i] = np.round(((hw[i] + (2 * dim_pad[i]) - dim_dil[i] * (dim_kernel[i] - 1) -1) / dim_stride[i]) + 1)
    output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    flops = counter_misc(np.prod(output_size) * np.prod(dim_kernel))
    # print(flops, output_size, output_name)
    module_logger.debug("Maxpool outtput_Size : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_globalaveragepool(diction, node):
    #flops = counter_zero_ops()
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    flops = counter_misc(np.prod(input_size))
    output_size = input_size
    return flops, output_size, output_name


def onnx_counter_concat(diction, node):
    #print("===========")
    #print(node)
    #print("===========")
    #print(diction)
    #print("===========")
    #print(diction[node.input[0]])
    axis = node.attribute[0].i
    input_size = diction[node.input[0]].copy()
    module_logger.debug("Input Size : %s", input_size)
    dim_concat = 0
    for i in node.input:
        dim_concat += diction[i][axis]
        module_logger.debug("i : channel size => %s : %s", i, diction[i][axis])
    output_size = input_size
    output_size[axis] = dim_concat
    output_name = node.output[0]
    #flops = counter_zero_ops()
    flops = counter_misc(np.prod(input_size))
    #print("onnx_counter_concat : ", flops)
    module_logger.debug("Output_Size : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_upsample(diction, node, scale):
    input_size = diction[node.input[0]].copy()
    module_logger.debug("Upsample input shape : %s", input_size)
    module_logger.debug("Node : %s", node)
    mode = node.attribute[0].s
    if not scale:
        scale = node.attribute[1].floats
        module_logger.debug("Scaling factor from scale attribute : %s", scale)
    module_logger.debug("Upsample scales, modes : %s, %s", scale, mode)
    module_logger.debug("Input 0 : %s", node.input[0])
    input_count = 0
    for i in node.input:
        input_count += 1
    if input_count > 1:
        module_logger.debug("Input 1 : %s", node.input[1])
    scale = np.array(scale)
    output_size = np.multiply(input_size, scale)
    output_size_scalar = np.prod(output_size)
    #flops = np.prod(input_size)
    flops = counter_upsample(mode, output_size_scalar)
    output_name = node.output[0]
    #print("scale factor : ", scale)  
    module_logger.debug("input_ size : %s", input_size)
    module_logger.debug("output size : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_clip(diction, node):
    flops = counter_zero_ops()
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    output_size = input_size
    return flops, output_size, output_name

def onnx_counter_dummy(diction, node):
    input_size = diction[node.input[0]].copy()
    acs = counter_zero_ops()
    output_name = node.output[0]
    output_size = input_size.copy()
    flops = 0.0
    return flops, output_size, output_name

def onnx_counter_squeeze(diction, node):
    # print(node)
    #flops = counter_zero_ops()
    output_name = node.output[0]
    axis = list(node.attribute[0].ints)
    module_logger.debug("%s : %s", axis, node.attribute[0].name)
    input_size = diction[node.input[0]].copy()
    module_logger.debug("input size of squeeze : %s, %s", input_size, len(input_size))
    output_size = []
    for i in range(len(input_size)):
        if isinstance(axis, list):
            if i not in axis:
                output_size.append(input_size[i])
        else:
            if axis != i:
                output_size.append(input_size[i])
        #print("i : ", i)
        #print("--->input size of squeeze ", input_size, len(input_size), i)
    if len(output_size) == 0:
        output_size = input_size
    else:
        output_size = np.array(output_size)
        
    flops = counter_misc(np.prod(input_size))
    module_logger.debug("output size of squeeze : %s, %s ", output_size, len(output_size))
    # print("flatten",output_size)
    return flops, output_size, output_name

def onnx_counter_unsqueeze(diction, node):
    # print(node)
    #flops = counter_zero_ops()
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    if node.attribute:
        axis = node.attribute[0].ints
        for i in axis:
            output_size = np.insert(input_size, i, 1)
        module_logger.debug("axis of unsqueeze : %s", axis)
    else:
        # Default behvior : unsqueeze to axe 0
        output_size = np.insert(input_size, 0, 1)
    module_logger.debug("input size of unsqueeze : %s, %s", input_size, len(input_size))
    flops = counter_misc(np.prod(input_size))
    module_logger.debug("output size of unsqueeze : %s, %s", output_size, len(output_size))
    # print("flatten",output_size)
    return flops, output_size, output_name

def onnx_counter_split(diction, node):
    axis = node.attribute[0].i
    split = node.attribute[1].ints
    input_size = diction[node.input[0]].copy()
    if axis < 0 :
        split_sum = 0
        for i in split:
            split_sum += i
        for i in range(len(input_size)):
            if split_sum == input_size[i]:
                axis = i
                
    input_size = diction[node.input[0]].copy()
    output_name_list = []
    output_size_list = []
    module_logger.debug("axis, split : %s, %s", axis, split)
    if (len(split) == 1):
        for i in range(len(node.output)):
            output_size = input_size
            input_size[i] = split[0]
    else:
        for i in range(len(node.output)):
            output_size = input_size
            output_size[axis] = split[i]
            output_size_list.append(output_size)
        
    for i in node.output:
        output_name_list.append(i)
    flops = counter_zero_ops()
    module_logger.debug("Split output size : %s, %s", i, output_size_list)
    return flops, output_size_list, output_name_list

def onnx_counter_transpose(diction, node):
    # print(node)
    #flops = counter_zero_ops()
    perms = node.attribute[0].ints
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    module_logger.debug("Transpose input_size : %s", input_size)
    module_logger.debug("Transpose permutation : %s", perms)
    # Ravi - Fixed
    output_size = input_size.copy() 
    for i in range(len(perms)):
        output_size[i] = input_size[perms[i]]
    flops = counter_misc(np.prod(input_size))
    #output_size = np.append(input_size[axis - 1], np.prod(input_size[axis:]))
    # print("flatten",output_size)
    module_logger.debug("Transpose output_size : %s", output_size)
    return flops, output_size, output_name

def onnx_counter_lstm(diction, node, param):
    module_logger.debug("LSTM node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    lstm_w_dim = param[0] 
    lstm_r_dim = param[1] 
    lstm_b_dim = param[2]
    num_direction = param[3]
    hidden_size = param[4]
    out_y_size = np.array([0,0,0,0])
    out_y_size[0] = input_size[0] # Sequence length
    out_y_size[1] = num_direction # Num direction
    out_y_size[2] = input_size[1] # Batch size
    out_y_size[3] = hidden_size   # Num sequence
    out_h_size = np.array([0,0,0])
    out_h_size[0] = num_direction
    out_h_size[1] = input_size[1]
    out_h_size[2] = hidden_size
    out_size = [out_y_size, out_h_size]
    #flops = counter_misc(np.prod(input_size))
    # Initial matrix multiplication
    flops = 4 * ((hidden_size * hidden_size) + (input_size[2] * hidden_size)) 
    # Count for sigmoid function
    flops += hidden_size * 3 * WEIGHT_SIGMOID
    # Count for tanh activation layer 
    flops += hidden_size * WEIGHT_TANH
    # Count for calculating cell state (1 MAC) 
    flops += hidden_size
    # Count for element-wise multiplication (Sigmoid output * ATAN output) 
    flops += hidden_size
    # Count for calculating final h and output y
    flops += hidden_size * (WEIGHT_TANH + 1)
    #flops = np.array([flops])
    # state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    # state_ops += hidden_size * 2
    # total_ops += state_ops * 4
    # # c' = f * c + i * g \\
    # # hadamard hadamard add
    # total_ops += hidden_size * 3
    # # h' = o * \tanh(c') \\
    # total_ops += hidden_size
    module_logger.debug("LSTM flops : %s", flops)
    module_logger.debug("LSTM node output shape: %s", out_size)
    module_logger.debug("LSTM node output name : %s", output_name)
    return flops, out_size, output_name

def onnx_counter_tile(diction, node, repeats):
    module_logger.debug("Tile node input : %s", diction[node.input[0]])
    output_name = node.output[0]
    input_size = diction[node.input[0]].copy()
    out_size = np.multiply(input_size, repeats)
    flops = counter_misc(np.prod(out_size))
    module_logger.debug("Tile node output : %s", out_size)
    return flops, out_size, output_name

def onnx_counter_nms(diction, node):
    # Ref:https://github.com/pytorch/vision/blob/96aa3d928c6faaf39defd846966421679244412d/torchvision/ops/boxes.py
    inputs_ = node.input
    input_tensor = []
    for inp_name in inputs_:
        input_tensor.append(diction.get(inp_name))
    boxes = input_tensor[0]
    if len(boxes) > 3:
        boxes = boxes[:-1]
    scores = input_tensor[1]
    flops = counter_nms(boxes[1], scores) 
    outsize = np.array([boxes[1], 3])
    output_name = node.output[0]
    return flops, outsize, output_name

def onnx_counter_tanh(diction, node):
    input_size = diction[node.input[0]].copy()
    outsize = input_size.copy()
    batch = input_size[0]
    output_name = node.output[0]
    nfeatures =  np.prod(input_size)
    flops = counter_tanh(batch, nfeatures)
    return flops, outsize, output_name

def onnx_counter_reduce_min(diction, node):
    attr_dict = parse_attribute(node)
    axes = attr_dict.get('axes', None)
    keep_dims = attr_dict.get('keepdims', 1)
    input_dim = diction.get(node.input[0])
    outsize = input_dim.copy()
    flops = 0
    if axes:
        for ax in list(axes):
            flops += input_dim[ax]
    else:
        flops += np.prod(input_dim)
    if bool(keep_dims):
        if axes:
            for ax in list(axes):
                outsize[ax] = 1
        else:
            for i in range(len(outsize)):
                outsize[i] = 1
    else:
        outsize = []
        if axes:
            for i in range(len(input_dim)):
                if i not in axes:
                    outsize.append(input_dim[i])
            outsize = np.array(outsize)
        else:
            outsize = np.array(outsize)
    output_name = node.output[0]
    return flops, outsize, output_name

onnx_operators = {
    "MatMul": onnx_counter_matmul,
    #"Add": onnx_counter_add,
    "Add": onnx_counter_add,
    "Conv": onnx_counter_conv,
    "Mul": onnx_counter_mul,
    #"Mul": onnx_counter_misc,
    "Constant": onnx_counter_constant,
    "BatchNormalization": onnx_counter_bn,
    "Relu": onnx_counter_relu,
    "ReduceMean": onnx_counter_reducemean,
    #"Sub": onnx_counter_sub,
    "Sub": onnx_counter_misc,
    "Pow": onnx_counter_pow,
    "Sqrt": onnx_counter_sqrt,
    "Div": onnx_counter_div,
    "InstanceNormalization": onnx_counter_instance,
    "Softmax": onnx_counter_softmax,
    "Pad": onnx_counter_pad,
    "AveragePool": onnx_counter_averagepool,
    "MaxPool": onnx_counter_maxpool,
    "Flatten": onnx_counter_flatten,
    "Gemm": onnx_counter_gemm,
    "GlobalAveragePool": onnx_counter_globalaveragepool,
    "Concat": onnx_counter_concat,
    "Clip": onnx_counter_clip,
    "Shape" : onnx_counter_misc,
    "Gather" : onnx_counter_gather,
    "Unsqueeze" : onnx_counter_unsqueeze,
    "Reshape" : onnx_counter_reshape,
    "MAdd" : onnx_counter_misc,
    "Exp" : onnx_counter_misc,
    "Split" : onnx_counter_misc,
    "Transpose" : onnx_counter_transpose,
    "View" : onnx_counter_misc,
    "Max" : onnx_counter_misc,
    "LRN" : onnx_counter_misc,
    "Elu" : onnx_counter_misc,
    "Upsample" : onnx_counter_upsample,
    "Sigmoid" : onnx_counter_misc,
    "ConvTranspose" : onnx_counter_conv_transpose,
    "PRelu" : onnx_counter_relu,
    "Squeeze" : onnx_counter_squeeze,
    "LeakyRelu" : onnx_counter_relu,
    "Dropout" : onnx_counter_dummy,
    "Identity" : onnx_counter_dummy,
    "Floor" : onnx_counter_misc,
    "Cast" : onnx_counter_misc,
    "GRU" : onnx_counter_misc,
    "LSTM" : onnx_counter_lstm,
    "Tile" : onnx_counter_tile,
    "Split" : onnx_counter_split,
    "Slice" : onnx_counter_slice,
    "Einsum" : onnx_counter_misc,
    "ConstantOfShape" : onnx_counter_misc,
    "Equal" : onnx_counter_misc,
    "Where" : onnx_counter_misc,
    "Expand" : onnx_counter_misc,
    "NonMaxSuppression": onnx_counter_nms,
    "ReduceMin": onnx_counter_reduce_min,
    "TopK": onnx_counter_topk,
    "Tanh": onnx_counter_tanh,
    None: None,
}

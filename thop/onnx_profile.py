import struct
import torch
import torch.nn
import onnx
import artifactory
from collections import OrderedDict
from onnx import numpy_helper
import numpy as np
from thop.vision.onnx_counter import onnx_operators
from google.protobuf.json_format import MessageToDict
import logging
import itertools
import pandas as pd
from pathlib import Path
from array import array
from .console import DisplaySummary
from rich.console import Console
# from rich import print
from rich.panel import Panel
from thop.utils import load_onnx, simplify_model, parse_attribute
from enum import Enum, unique
from typing import Tuple
import google.protobuf.pyext._message as message 
from onnx.onnx_ml_pb2 import NodeProto
from rich.table import Table

@unique
class FlopsOpt(Enum):
    KILO = ('K', 1e3) # Units to be displayed in Kilo (K) and its corresponding value 
    MEGA = ('M', 1e6) # Units to be displayed in Mega (M) and its corresponding value 
    GIGA = ('G', 1e9) # Units to be displayed in Giga (G) and its corresponding value 

    def describe(self):
        return self.name, self.value

    def __str__(self):         
        return self.value[0]
class OnnxProfile(DisplaySummary):
    '''
    Class OnnxProfile is used to profile the onnx files.

    This class calculates the aggregated FLOPs and parameters for each layer in the onnx graph.
    It also provide methods to display the onnx graph layers in and out dimension
    along with flops and parameters for the layer. 

    Parameters
    ----------
    DisplaySummary : abc.ABCMeta
        This is a base class for displaying the statistics collected for each layer
        of the model graph in form of table on a Console. This class has abstract
        method for setting the header for the table and a method for generating the
        table from the collected stats.
    '''
    def __init__(self, flops_opt: FlopsOpt = FlopsOpt.KILO):
        '''
        This is a __init__ method for Class OnnxProfile.

        This method initials the necessary variables for displaying the stats in 
        appropriate units.

        Parameters
        ----------
        flops_opt : FlopsOpt, optional
            The user has a choice to display the calculated FLOPs in appropriate 
            unit like KILO, MEGA or GIGA, by default FlopsOpt.KILO
        '''
        DisplaySummary.__init__(self)
        self._flops_opt = flops_opt
        self.console = Console(record=True)
        self._op_count_dict= {}
        self._layer_stat = []
        self._aggr_dict = {}
        self._name2dims: OrderedDict[str, array] = OrderedDict()
        self.logger = logging.getLogger(f'onnx_profiling.{str(Path(__file__).stem)}.OnnxProfile')

    # getting the values
    @property
    def flops_opt(self):
        '''
        This is a flops_opt get method

        This is getter method for getting the value of _flops_opt variable 
        which has the information about the unit in which the calculated
        FLOPs will be displayed.

        Returns
        -------
        FlopsOpt
            The value returned is FlopsOpt ENUM value. The return value can 
            be any of these FlopsOpt.KILO, FlopsOpt.MEGA or FlopsOpt.GIGA.
        '''
        self.logger.info('Getting value')
        return self._flops_opt
 
    # setting the values
    @flops_opt.setter
    def flops_opt(self, flops_opt):
        '''
        This is a flops_opt set method

        This is setter method for setting the _flops_opt variable with value
        which has the information about the unit in which the calculated
        FLOPs will be displayed.

        Parameters
        ----------
        flops_opt : FlopsOpt
            The value set is FlopsOpt ENUM value. This value can 
            be any of these FlopsOpt.KILO, FlopsOpt.MEGA or FlopsOpt.GIGA.
        '''
        self.logger.info('Setting value to %s', flops_opt)
        self._flops_opt = flops_opt

    def _calculate_total_params(self, model: onnx.ModelProto):
        '''
        This is a private _calculate_total_params method.

        This method is used for calculating the total number of parameters for
        a given onnx graph. It goes over each layer weights and calculates the
        total parameters for that layer, and later aggregates over the entire
        onnx graph.

        Parameters
        ----------
        model : onnx.ModelProto
            This is a onnx graph which is loaded using `onnx.load`.

        Returns
        -------
        ndarray
            This is the aggregated number of parameters for a given onnx.ModelProto.
        '''
        onnx_weights = model.graph.initializer
        params = 0

        for onnx_w in onnx_weights:
            try:
                weight = numpy_helper.to_array(onnx_w)
                params += np.prod(weight.shape)
            except Exception as _:
                pass

        return params

    def _calculate_layer_params(self, model: onnx.ModelProto, layer_param_name: list):
        '''
        This is a private _calculate_layer_params method.

        This method is used for calculating the total number of parameters for each
        layer in the given onnx graph.

        Parameters
        ----------
        model : onnx.ModelProto
            This is a onnx graph which is loaded using `onnx.load`.
        layer_param_name : list
            This is list of parameter name present in the layer for which the total
            number of parameters is calculated.

        Returns
        -------
        ndarray
            This is the aggregated number of parameters for a given list of parameter 
            names for a given layer.
        '''
        onnx_weights = model.graph.initializer
        params = 0
        for param_name in layer_param_name:
            onnx_w_list = list(filter(lambda w: w.name == param_name, onnx_weights))
            if onnx_w_list:
                onnx_w = onnx_w_list[0]
                try:
                    weight = numpy_helper.to_array(onnx_w)
                    params += np.prod(weight.shape)
                except Exception as _:
                    pass
            else:
                param = self._name2dims.get(param_name, None)
                if param is not None:
                    params += np.prod(self._name2dims.get(param_name).shape) 
                else:
                    params += 0
        return params

    def _create_dict(self, weight: message.RepeatedCompositeContainer, 
    input: message.RepeatedCompositeContainer, output: message.RepeatedCompositeContainer,
    nodes: message.RepeatedCompositeContainer) -> None:
        '''
        This is a private _create_dict method.

        This method updates the dictionary with key being the name of the input tensor, 
        output tensor, the weight parameter name and the value being the corresponding 
        shape tensor.  

        Parameters
        ----------
        weight : message.RepeatedCompositeContainer
            This container containes the list of onnx weight tensor.
        input : message.RepeatedCompositeContainer
            This container containes the list of onnx input tensor.
        output : message.RepeatedCompositeContainer
            This container containes the list of onnx output tensor.
        '''
        for n in nodes:
            if n.op_type == 'Constant':
                attr = n.attribute[0].t
                self._name2dims[str(n.name)] = np.absolute(attr.dims)
        for w in weight:
            dim = np.array(w.dims)
            self._name2dims[str(w.name)] = np.absolute(dim)
            # if dim.size == 1:
            #     self._name2dims[str(w.name)] = np.append(1, dim)
        for i in input:
            ## Ravi - Fixed
            m_dict = MessageToDict(i)
            if m_dict:
                dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
                if dim_info:
                    input_shape = [int(d.get("dimValue")) if 'dimValue' in d else 0 for d in dim_info ]
                    input_shape = np.absolute(input_shape)
                    self._name2dims[str(i.name)] = input_shape
                    if input_shape.size == 1:
                        self._name2dims[str(i.name)] = np.append(1, input_shape)
        for o in output:
            ## Ravi - Fixed
            m_dict = MessageToDict(o)
            if m_dict:
                dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
                if dim_info:
                    output_shape = [int(d.get("dimValue")) if 'dimValue' in d else 0 for d in dim_info]
                    output_shape = np.absolute(output_shape)
                    self._name2dims[str(o.name)] = output_shape
                    if output_shape.size == 1:
                        self._name2dims[str(o.name)] = np.append(1, output_shape)

    def nodes_counter(self, node: NodeProto, param = [])-> Tuple[torch.Tensor, np.ndarray, str]:
        '''
        This is a nodes_counter method.

        This method maps the node type to appropriate node counter funtion. 

        Parameters
        ----------
        node : NodeProto
            This is a layer in a onnx graph which has all the information 
            need to compute the FLOPs.
        param : list, optional
            This is a list for weight parameters that is present for the 
            layer that is being processed , by default []
        Returns
        -------
        Tuple[torch.Tensor, np.ndarray, str]
            These are the outputs from the node operation counter. The outputs 
            are total FLOPs for the layer, output size of the tensor for that
            layer, output name of the tensor for that layer.
        '''
        diction = self._name2dims.copy()
        if node.op_type not in onnx_operators:
            self.logger.error("Sorry, we haven't add %s into dictionary.", node.op_type)
            return 0, np.array([]), None
        elif node.op_type == 'Reshape':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'Upsample':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'LSTM':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'Tile':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'Slice':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'Gather':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        elif node.op_type == 'TopK':
            fn = onnx_operators[node.op_type]
            return fn(diction, node, param)
        else:
            fn = onnx_operators[node.op_type]
            return fn(diction, node)

    def find_constant_node(self, input_name: str, nodes: message.RepeatedCompositeContainer) -> Tuple or message.RepeatedScalarContainer:
        '''
        This is find_constant_node method.

        This method extracts the Constant tensor for a specified node name.

        Parameters
        ----------
        input_name : str
            This is the input node name for which we need to extract Constant Tensor Value.
        nodes : message.RepeatedCompositeContainer
            This are the nodes from the model graph.

        Returns
        -------
        Tuple | message.RepeatedScalarContainer
            The output of this method is a Constant Tensor values
        '''
        for n in nodes:
            if n.output[0] == input_name and n.op_type == 'Constant':
                attr = n.attribute[0].t
                self.logger.debug("Raw Data : %s" , attr.raw_data)
                if attr.raw_data:
                    self.logger.debug("Len : %s", attr.dims)
                    fmt = ''
                    if len(attr.dims) != 0:
                        for i in range(attr.dims[0]):
                            if attr.data_type == 1:
                                fmt += 'f'
                            elif attr.data_type == 7:
                                fmt += 'q'
                        self.logger.debug("Format : %s", fmt)
                        constants = struct.unpack(fmt, attr.raw_data)
                        return constants
                    else:
                        if attr.data_type == 1:
                            fmt += 'f'
                        elif attr.data_type == 7:
                            fmt += 'q'
                        self.logger.debug("Format : %s", fmt)
                        constants = struct.unpack(fmt, attr.raw_data)
                        return constants
                else:
                    if attr.data_type == 1:
                        constants = attr.float_data
                    elif attr.data_type == 7:
                        constants = attr.int64_data
                    return constants

    def find_constant_weight(self, input_name: str, weight: message.RepeatedCompositeContainer) -> Tuple or message.RepeatedScalarContainer:
        '''
        This is find_constant_weight method.

        This method extracts the Constant Weight tensor for a specified node name.


        Parameters
        ----------
        input_name : str
            This is the input node name for which we need to extract Constant Weight Tensor Value.
        weight : message.RepeatedCompositeContainer
            This are the node weights from the model graph.

        Returns
        -------
        Tuple or message.RepeatedScalarContainer
            The output of this method is a Constant Weight Tensor values
        '''
        for w in weight:
            if w.name == input_name:
                self.logger.debug("w : %s" , w)
                self.logger.debug("Raw Data %s: " , w.raw_data)
                if w.raw_data:
                    self.logger.debug("Len : %s", w.dims)
                    self.logger.debug("data_type : %s", w.data_type)
                    fmt = ''
                    fmt_len = int(w.dims[0]) if w.dims else 0
                    for i in range(fmt_len): 
                        if w.data_type == 1:
                            fmt += 'f'
                        elif w.data_type == 7:
                            fmt += 'q'
                    
                    if fmt_len == 0:
                        if w.data_type == 1:
                            fmt += 'f'
                        elif w.data_type == 7:
                            fmt += 'q'
                    
                    self.logger.debug("Format : %s", fmt)
                    constants = struct.unpack(fmt, w.raw_data)
                    return constants
                else:
                    if w.data_type == 1:
                        constants = w.floats
                    elif w.data_type == 7:
                        constants = w.int64_data
                    return constants

    def find_dimension_weight(self, input_name: str, weight: message.RepeatedCompositeContainer) -> np.array:
        '''
        This is a find_dimension_weight method.

        This method to calculate the dimension of the weight tensor
        for a specified node name.

        Parameters
        ----------
        input_name : str
            This is the input node name for which we need to calculate
            the weight dimension.
        weight : message.RepeatedCompositeContainer
            This are the node weights from the model graph.

        Returns
        -------
        np.array
            This is the dimension of the specified weight tensor.
        '''
        for w in weight:
            if w.name == input_name:
                self.logger.debug("w.dims : %s" , w.dims)
                return np.array(w.dims)

    def _parse_io(self, model: onnx.ModelProto) -> None:
        '''
        This is private _parse_io method.

        This method is used to parse the shape tensor from 
        input, output and weights of the onnx graph.

        Parameters
        ----------
        model : onnx.ModelProto
            This is a onnx graph which is loaded using `onnx.load`.
        '''
        self._name2dims = OrderedDict()
        weight = model.graph.initializer
        nodes = model.graph.node
        input = model.graph.input
        output = model.graph.output
        self._create_dict(weight, input, output, nodes)
        # Sometimes node[1] instead of node[0] is input 
        # if nodes[0].input:
        #     global_input = self._name2dims[nodes[0].input[0]]
        #     model_input = nodes[0].input[0]
        # else:
        #     global_input = self._name2dims[nodes[1].input[0]]
        #     model_input = nodes[1].input[0]
        # #print("============ Global Input", global_input)
        # # Exception handling : Input shape  [A, B, C, D] C, D are assumed for 2d input size
        # # B is the number of input channles
        # # Some models assume D for the number of input channes, and B, C for 2D input size 
        # if (global_input[-1] < global_input[-2]) and (global_input[-1] < global_input[-3]):
        #     global_input = np.insert(global_input, 1, global_input[-1])
        #     global_input = np.delete(global_input, global_input.size-1)
        #     self.logger.debug("Got cha! %s", global_input)
        # self._name2dims[model_input] = global_input

    @staticmethod
    def _convert_str_format(array: np.ndarray) -> str:
        '''
        This is static _convert_str_format method.

        This method convert the numpy array value into a str.
        
        Example:
            np.array([1,3,224,224]) -> 1x3x224x224

        Parameters
        ----------
        array : np.ndarray
            This is numpy array which represents the shape tensor 
            of the input output layers of onnx graph.

        Returns
        -------
        str
            The string of shape tensor in num1xnum2xnum3xnum4 format.
        '''
        return 'x'.join(list(map(lambda i: str(int(i)), array)))

    def _push_stats(self,  model: onnx.ModelProto, node: onnx.NodeProto, layer_macs: torch.DoubleTensor) -> None:
        '''
        This is private _push_stats method.

        This method keeps track of the Name, Type, Input/Output Tensor Dimension, FLOPs and Parameters
        for each layer in the onnx graph. This table is displayed in the Console as a detailed summary.

        Parameters
        ----------
        model : onnx.ModelProto
            This is a onnx graph which is loaded using `onnx.load`.
        node : onnx.NodeProto
            This is a layer in a onnx graph which has all the information about the node.
        layer_macs : torch.DoubleTensor
            This is the total FLOPs for the node in the onnx graph.
        '''
        stats = {'Layer': node.op_type, 'Name': None if node.name == '' else node.name}
        try:
            if node.input:
                if node.op_type == 'Concat' or node.op_type == 'MatMul':
                    stats['Input Dim'] = ', '.join(list(map(lambda x: self._convert_str_format(self._name2dims.get(x, None)), node.input)))
                    # stats['Input Dim'] = f'{self._convert_str_format(self._name2dims.get(node.input[0], None))}, {self._convert_str_format(self._name2dims.get(node.input[1], None))}'
                else:
                    stats['Input Dim'] = f'{self._convert_str_format(self._name2dims.get(node.input[0], None))}'
            else:
                stats['Input Dim'] = None
        except Exception as e:
            stats['Input Dim'] = None
        try:
            if node.output:
                if node.op_type == 'Constant':
                    stats['Output Dim'] = None
                elif node.op_type == 'Dropout':
                    stats['Output Dim'] = self._convert_str_format(self._name2dims.get(node.output[0], None))
                else:
                    stats['Output Dim'] = ' , '.join(list(map(lambda o: self._convert_str_format(self._name2dims.get(o, None)), node.output)))
            else:
                stats['Output Dim'] = None
        except Exception as e:
            stats['Output Dim'] = None
        stats['FLOPS'] = layer_macs.item() if isinstance(layer_macs, torch.Tensor) else layer_macs
        if node.op_type == 'Concat':
            stats['Parameters'] = self._calculate_layer_params(model, node.input[2:])
        else:
            stats['Parameters'] = self._calculate_layer_params(model, node.input[1:])
        self._layer_stat.append(stats)

    def calculate_macs(self, model: onnx.ModelProto) -> torch.DoubleTensor:
        '''
        This is public calculate_macs method.

        This method takes the onnx graph and calculates the FLOPs for each layer and other
        necessary information about the layer, along with the aggregated information of 
        the entire onnx graph.

        Parameters
        ----------
        model : onnx.ModelProto
            This is a onnx graph which is loaded using `onnx.load`.

        Returns
        -------
        torch.DoubleTensor
            This is the total FLOPs for a given onnx graph.
        '''
        macs = 0
        self._op_count_dict = {}
        self._layer_stat = []
        self._parse_io(model)
        for n in model.graph.node:
            self.logger.debug("Operation type: %s", n.op_type)
            if (n.op_type == 'Constant'):
                continue
            elif (n.op_type == 'Split'):
                macs_adding, out_size_list, outname_list = self.nodes_counter(n)
                for i in range(len(out_size_list)):
                    out_size = np.absolute(out_size_list[i])
                    self._name2dims[outname_list[i]] = out_size
                self.logger.debug("Dict : %s", self._name2dims)
            # Case 1 : Reshape has parameter from another output -> reshape = []
            # Case 2 : Reshape has constant parameter (parameter in constant node or initializer (weight))
            elif (n.op_type == 'Reshape'):
                if n.attribute:
                    reshape_shape = parse_attribute(n).get('shape')
                else:
                    input_name = n.input[1]
                    input_count = 0
                    self.logger.debug("Reshape input_name, input_count : %s, %s", input_name, input_count)
                    reshape_shape = self.find_constant_node(input_name, model.graph.node)
                    self.logger.debug("Reshape_shape from constant : %s", reshape_shape)
                    if (not reshape_shape):
                        reshape_shape = self.find_constant_weight(input_name, model.graph.initializer)
                self.logger.debug("Reshape_shape from weight : %s", reshape_shape)
                self.logger.debug("Reshape_shape from constant : %s", reshape_shape)
                macs_adding, out_size, outname = self.nodes_counter(n, reshape_shape)
                out_size = np.absolute(out_size)
                self._name2dims[outname] = out_size
                
            # Upsample scale factor 
            # Case 1 : from initializer (weights)
            # Case 2 : from output of constant node (nodes)
            # Case 3 : from an output shape of another node
            elif (n.op_type == 'Upsample'):
                input_count = 0
                scale = []
                for i in n.input:
                    input_count += 1
                if (input_count == 2):
                    input_name = n.input[1]
                    self.logger.debug("Scale input_name : %s", input_name)
                    scale = self.find_constant_weight(input_name, model.graph.initializer)
                    if scale:
                        self.logger.debug("Scale from weight : %s", scale)
                    else:
                        scale = self.find_constant_node(input_name, model.graph.node)
                        if scale :
                            self.logger.debug("Scale from node : %s", scale)
                        else:
                            scale = self._name2dims[input_name]
                            self.logger.debug("Scale from dict : %s", scale)
                macs_adding, out_size, outname = self.nodes_counter(n, scale)
                out_size = np.absolute(out_size)
                self._name2dims[outname] = out_size
            elif n.op_type == 'LSTM':
                self.logger.debug("LSTM Input : %s", n.input)
                lstm_b = n.input[3]
                lstm_r = n.input[2]
                lstm_w = n.input[1]
                #output_szie = self.find_constant_node(bias_name, nodes)
                self.logger.debug("Name lstm_w, lstm_r, lstm_b : %s, %s, %s", lstm_w, lstm_r, lstm_b)
                self.logger.debug("LSTM attribute : %s", n.attribute)
                direction = 1
                for attr in n.attribute:
                # print(attr)
                    if attr.name == "hidden_size":
                        hidden_size = attr.i 
                        self.logger.debug("Hidden size %s", hidden_size)
                    if attr.name == "direction":
                        if attr.s.decode('UTF-8') == 'bidirectional':
                            direction = 2
                        else:
                            direction = 1
                lstm_w_dim = self.find_dimension_weight(lstm_w, model.graph.initializer)
                lstm_r_dim = self.find_dimension_weight(lstm_r, model.graph.initializer)
                lstm_b_dim = self.find_dimension_weight(lstm_b, model.graph.initializer)
                params = [lstm_w_dim, lstm_r_dim, lstm_b_dim, direction, hidden_size]
                self.logger.debug("LSTM Parameters : %s", params)
                macs_adding, out_size, outname = self.nodes_counter(n, params)
                self.logger.debug("LSTM macs_adding : %s", macs_adding)
                self._name2dims[outname] = out_size [0]
                self._name2dims[n.output[1]] = out_size [1]
                self._name2dims[n.output[2]] = out_size [1]
            elif n.op_type == 'Tile':
                repeats_name = n.input[1]
                repeats = self.find_constant_weight(repeats_name, model.graph.initializer)
                self.logger.debug("Tile repeats name: %s", repeats_name)
                macs_adding, out_size, outname = self.nodes_counter(n, repeats)
                self.logger.debug("Tile repeats : %s", repeats)
                self._name2dims[outname] = out_size
            elif n.op_type == 'Gather':
                input_params = list(map(lambda x: (x, self.find_constant_weight(x, model.graph.initializer)[0] if self.find_constant_weight(x, model.graph.initializer) else self.find_constant_node(x, model.graph.node)[0]), n.input[1:]))
                macs_adding, out_size, outname = self.nodes_counter(n, input_params)
                self.logger.debug("out_size, outname : %s, %s", out_size, outname)
                out_size = np.absolute(out_size)
                self._name2dims[outname] = out_size
            elif n.op_type == 'Slice':
                input_params = list(map(lambda x: (x, self.find_constant_weight(x, model.graph.initializer)[0]), n.input[1:]))
                macs_adding, out_size, outname = self.nodes_counter(n, input_params)
                self.logger.debug("out_size, outname : %s, %s", out_size, outname)
                out_size = np.absolute(out_size)
                self._name2dims[outname] = out_size
            elif n.op_type == 'TopK':
                input_params = list(map(lambda x: (x, self.find_constant_weight(x, model.graph.initializer)[0] if self.find_constant_weight(x, model.graph.initializer) else 1), n.input[1:]))
                macs_adding, out_size, outname = self.nodes_counter(n, input_params)
                self.logger.debug("out_size, outname : %s, %s", out_size, outname)
                out_size = list(map(lambda x: np.absolute(x), out_size))
                for name, value in zip(outname, out_size):
                    self._name2dims[name] = value
            else:
                macs_adding, out_size, outname = self.nodes_counter(n)
                self.logger.debug("out_size, outname : %s, %s", out_size, outname)
                out_size = np.absolute(out_size)
                self._name2dims[outname] = out_size
            macs += float(macs_adding)
            self.logger.debug("input, output  : %s, %s", n.input, n.output)
            self.logger.debug("macs, op_type  : %s, %s", macs, n.op_type)
            self.logger.debug("macs_adding, macs  : %s, %s", macs_adding, macs)
            if (n.op_type in self._op_count_dict):
                self._op_count_dict[n.op_type] += float(macs_adding)
            else :
                self._op_count_dict[n.op_type] = float(macs_adding)
            
            self._push_stats(model, n, macs_adding)
        return macs

    def calculate_macs_agg(self, model_path: str or artifactory.ArtifactoryPath, simplify: bool=False) -> None:
        '''
        This is public calculate_macs_agg method.

        This method calculates the aggregated FLOPs for each model that is present
        in the regression list. It also keeps track of aggregated statistics for each
        model and the summary of these regression models are exported. 

        Parameters
        ----------
        model_path : str or artifactory.ArtifactoryPath
            This is the file path to onnx model. we can also pass boartifactory 
            ArtifactoryPath to load the onnx model from cloud. 
        simplify : bool, optional
            This flag will determine if the onnx model needs to simplified before
            calculating the aggregated statistics, by default False
        '''
        self.logger.debug("Model Name: %s", Path(model_path).stem)
        model_onnx_path = None
        if isinstance(model_path, str):
            model_onnx_path = model_path
        elif isinstance(model_path, artifactory.ArtifactoryPath):
            with model_path.open() as fd:
                model_onnx_path = fd.read()
        try:
            model = load_onnx(model_path=model_onnx_path)
            if simplify:
                model = simplify_model(model)
            _ = self.calculate_macs(model)
            self._aggr_dict[Path(model_path).stem] = self._op_count_dict
        except Exception as e:
            print(Path(model_path).stem)
            self.logger.error("Exception Occurs! : %s", e, exc_info=True)

    def _set_header(self) -> None:
        '''
        This is an abstract _set_header method.

        This method will allow to the user to set appropriate Table Headers for 
        displaying the stats of each layer of the onnx graph on the console.
        '''
        self.display_logger.debug('Setting the Table Header.')
        self.table.add_column('[bright_yellow]Layer', justify='center')
        self.table.add_column('[bright_yellow]Name', justify='center')
        self.table.add_column('[bright_yellow]Input Dim', justify='center')
        self.table.add_column('[bright_yellow]Output Dim', justify='center')
        self.table.add_column(f'[bright_yellow]FLOPs({self._flops_opt.describe()[1][0]})', justify='center')
        self.table.add_column('[bright_yellow]Parameters', justify='center')
        self.table.add_column(f'[bright_yellow]FLOPs({self._flops_opt.describe()[1][0]}) (in %)', justify='center')
        self.table.add_column('[bright_yellow]Parameters (in %)', justify='center')

    def generate_table(self, table_name: str, round_decimal: int=2) -> Table:
        '''
        This is an abstract generate_table method.

        This method creates the Table with appropriate Table name and the headers. 
        It also updates the rows of the table (which represent a layer in onnx graph) 
        with appropriate stats.

        Parameters
        ----------
        table_name : str
            The is the table name to be displayed on top of the table.
        round_decimal : int, optional
            This variable will determine upto which decimal value the stats needs 
            to rounded off before displaying, by default 2

        Returns
        -------
        Table
            This is the updated Table object with all the stats for a given onnx graph.
        '''
        self.table_name = table_name
        self.display_logger.debug('Table Name : %s', self.table_name)
        self._create_table()
        _total_params = sum(list(map(lambda x: float(x.get('Parameters', 0)), self._layer_stat)))
        _total_flops = sum(list(map(lambda x: float(x.get('FLOPS', 0)), self._layer_stat)))
        if len(self._layer_stat) > 0:
            for row_dict in self._layer_stat:
                self.table.add_row(
                    f"{row_dict.get('Layer', '')}", f"{row_dict.get('Name', '')}", f"{row_dict.get('Input Dim', '')}",
                    f"{row_dict.get('Output Dim', '')}", f"{ (row_dict.get('FLOPS', '') / self._flops_opt.describe()[1][1]):.{round_decimal}f}", 
                    f"{row_dict.get('Parameters', '')}",
                    f"{(row_dict.get('FLOPS', '') / _total_flops) * 100 :.{round_decimal}f}",
                    f"{(row_dict.get('Parameters', '') / _total_params) * 100:.{round_decimal}f}"
                )
        return self.table

    def overall_stat(self, model_name: str, round_decimal=2) -> None:
        '''
        This is a public overall_stat method.

        This method calculate the overall Statistic for a given onnx graph and displays them
        on the console. The Statistics displayed are the Total Parameters and the Total FLOPs. 

        Parameters
        ----------
        model_name : str
            This is the model name to be displayed on the Console with the overall stats.
        round_decimal : int, optional
            This variable will determine upto which decimal value the stats needs 
            to rounded off before displaying, by default 2
        '''
        _total_params = sum(list(map(lambda x: float(x.get('Parameters', 0)), self._layer_stat)))
        _total_flops = sum(list(map(lambda x: float(x.get('FLOPS', 0)), self._layer_stat)))

        self.console.print(Panel.fit(f' [cyan]Overall Stats for Model: {model_name} \n [green]Total Parameters: {_total_params:.{round_decimal}f} \n [yellow]Total FLOPs ({self._flops_opt.describe()[1][0]}): {( _total_flops/ self._flops_opt.describe()[1][1]):.{round_decimal}f}'))

    def export_to_csv(self, round_decimal=2, save_path='regression_flops_output_local.xlsx') -> None:
        '''
        This is a public export_to_csv method.

        This method with allow the user to export the overall Floating point operations for each 
        models in the regression list into a .xlsx format with sheet named Ops. This method also 
        creates another sheet named Ops(in %) which as the overall Floating point operations in
        percentage format for each models in the regression list. This would give you an idea of 
        what percentage of Operation are significantly present in the model.

        Parameters
        ----------
        round_decimal : int, optional
            This variable will determine upto which decimal value the stats needs 
            to rounded off before displaying, by default 2
        save_path : str, optional
            This is the path where the .xlsx file will be saved, by default 'regression_flops_output_local.xlsx'
        '''
        collected_op_types = sorted(list(set(itertools.chain.from_iterable([ list(self._aggr_dict[key].keys()) for key in self._aggr_dict.keys()]))))
        data = {}
        if bool(self._aggr_dict):
            for model_name in self._aggr_dict.keys():
                if 'Models' not in data:
                    data['Models'] = [model_name]
                else:
                    data['Models'].append(model_name)
                for op_type in collected_op_types:
                    if op_type not in data:
                        data[op_type] = [self._aggr_dict[model_name].get(op_type, 0)]
                    else:
                        data[op_type].append(self._aggr_dict[model_name].get(op_type, 0))
            df = pd.DataFrame(data=data)
            df.set_index('Models')
            df_perc = df.copy(deep=True)
            writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Ops')
            df_perc[collected_op_types] = df_perc[collected_op_types].div(df_perc[collected_op_types].sum(axis=1), axis=0) * 100
            if round_decimal and isinstance(round_decimal, int):
                df_perc[collected_op_types] = df_perc[collected_op_types].round(round_decimal)
            df_perc.to_excel(writer, sheet_name='Ops (%)')
            writer.save()







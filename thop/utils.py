import onnx
import itertools
import logging
from pathlib import Path
from onnxsim import simplify
from typing import List
from collections import OrderedDict
from google.protobuf.json_format import MessageToDict
import numpy as np

module_logger = logging.getLogger(f'onnx_profiling.onnx_profile.{str(Path(__file__).stem)}')


def load_onnx(model_path: str or bytes) -> onnx.onnx_dot_onnx__ml__pb2.ModelProto:
    try:
        if isinstance(model_path, str):
            return onnx.load(model_path)
        elif isinstance(model_path, bytes):
            return  onnx.load_from_string(model_path)
    except Exception as e:
        module_logger.error("Exception Occurs! : %s", e, exc_info=True)

def parse_input(onnx_model: onnx.onnx_dot_onnx__ml__pb2.ModelProto) -> OrderedDict:
        inputs = onnx_model.graph.input
        input_name_list = [node.name for node in onnx_model.graph.input]
        input_initializer_name_list =  [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_name_list)  - set(input_initializer_name_list))
        input_dict: OrderedDict[str, str] = OrderedDict()
        for inp in inputs:
            if inp.name in net_feed_input:
                m_dict = MessageToDict(inp)
                if m_dict:
                    dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
                    if dim_info:
                        input_shape = [int(d.get("dimValue")) if 'dimValue' in d else (1 if 'dimParam' in d else 0)for d in dim_info ]
                        input_shape = np.absolute(input_shape)
                        input_dict[str(inp.name)] = ','.join(list(map(str, input_shape)))
        return input_dict

def simplify_model(onnx_model: onnx.onnx_dot_onnx__ml__pb2.ModelProto) -> onnx.onnx_dot_onnx__ml__pb2.ModelProto:

        input_dict = parse_input(onnx_model=onnx_model)
        # convert model
        input_shapes = dict()
        if bool(input_dict):
            for inp in input_dict.keys():
                x = f'{inp}:{input_dict[inp]}'
                if ':' not in x:
                    input_shapes[None] = list(map(int, x.split(',')))
                else:
                    pieces = x.split(':')
                    # for the input name like input:0
                    name, shape = ':'.join(
                        pieces[:-1]), list(map(int, pieces[-1].split(',')))
                    input_shapes.update({name: shape})
        try:
            onnx_model_simp, _ = simplify(onnx_model, input_shapes=input_shapes)
        
        except Exception as e:
            module_logger.error("Exception Occurs! : %s", e, exc_info=True)
        return onnx_model_simp

class OnnxLayerSize(object):

    def __init__(self) -> None:
        self.layer_size = OrderedDict()

    def _calculate_layer_size(self, shape_proto: List) -> None:
        for shape in shape_proto:
            m_dict = MessageToDict(shape)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            if dim_info:
                input_shape = [int(d.get("dimValue")) if 'dimValue' in d else 0 for d in dim_info ]
                input_shape = np.absolute(input_shape)
                self.layer_size[m_dict.get("name")] = 'x'.join(list(map(str, input_shape)))

    def _fetch_shape_proto(self, onnx_model: onnx.onnx_dot_onnx__ml__pb2.ModelProto) -> List:
        inter_layers = list(itertools.chain.from_iterable([node.output for node in onnx_model.graph.node]))# output tensor names
        # output_layers = list(map(lambda x: x.name, onnx_model.graph.output))
        # filter_inter_layers = list(filter(lambda x: x not in output_layers, inter_layers))
        value_info_protos = list()
        shape_info = onnx.shape_inference.infer_shapes(onnx_model)
        for _, node in enumerate(shape_info.graph.value_info):
            if node.name in inter_layers:
                value_info_protos.append(node)
        value_info_protos.extend(onnx_model.graph.output)
        assert len(value_info_protos) == len(inter_layers)
        return value_info_protos

    def calculate_onnx_layers_io_size(self, onnx_model: onnx.onnx_dot_onnx__ml__pb2.ModelProto, simplify=True):
        if simplify:
            onnx_model = simplify_model(onnx_model=onnx_model)
        shape_proto = self._fetch_shape_proto(onnx_model=onnx_model)
        self._calculate_layer_size(shape_proto=shape_proto)
        return onnx_model

def parse_attribute(node: onnx.NodeProto) -> dict:
    attr_dict = dict()
    for attr in node.attribute:
        attr_dict[attr.name] = attr.i if attr.ints == [] else attr.ints
    return attr_dict

def pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    num_axes = len(axes)

    if num_axes * 2 != raw_pads.size:
        raise Exception("The number of elements in raw_pads should be 2 * num_axes")

    pad_width = []
    for i in range(input_rank):
        pad_width += [[0, 0]]  # init to zero

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    for i in range(num_axes):
        axis = axes[i]
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]

    if mode == "constant":
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y


if __name__ == '__main__':

    o_OLS = OnnxLayerSize()
    model = onnx.load('/Users/raviraagavsr/Data/model/alexnet.onnx')
    o_OLS.calculate_onnx_layers_io_size(model)

from enum import Enum
from typing import Any, List, Optional, Tuple

class Operators(Enum): 
    INPUT = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MATMUL = 5
    RELU = 6
    LAYERNORM = 7
    SIGMOID = 8
    TANH = 9
    SOFTMAX_CROSS_ENTROPY = 10
    CONV2D = 11
    MAXPOOL2D = 12
    AVGPOOL2D = 13
    FLATTEN = 14
    DENSE = 15
    DROPOUT = 16
    BATCHNORM = 17
    RESHAPE = 18
    CONCAT = 19
    SPLIT = 20   
    TRANSPOSE = 21
    SUM = 22
    MEAN = 23
    VARIANCE = 24
    ARGMAX = 25
    ARGMIN = 26
    EXP = 27
    LOG = 28
    SQRT = 29
    POWER = 30
    CLIP = 31
    SLICE = 32
    GATHER = 33
    SCATTER = 34
    BROADCAST = 35
    PAD = 36
    NORMALIZE = 37
    LSTM = 38
    GRU = 39
    MHA = 40
    GELU = 41
    FFN = 42
    CUSTOM = 99
class Node: 
    def __init__(self, op_type: Operators, inputs: Optional[List['Node']] = None, attributes: Optional[dict] = None, name: Optional[str]= None, shape: Tuple[int, ...]=None, requires_grad: bool = False, backward_func=None, saved_tensors: List=None) -> None:
        self.op_type = op_type
        self.inputs = inputs if inputs is not None else []
        self.attributes = attributes if attributes is not None else {}
        self.name = name if name is not None else f"{op_type.name}_node"
        self.id = id(self)
        self.shape = shape if shape is not None else ()
        self.requires_grad = requires_grad 
        self.grad = None           # To be allocated: 400MB for 100M model
        self.backward_fn = None    # The chain rule implementation
        self.saved_tensors = []
        self.data = None

    def __repr__(self) -> str:
        base_name = f"Node(id={self.id}), shape = {self.shape}"
        if self.name: 
            base_name += f", Name={self.name}"
        if self.inputs: 
            input_names = [str(i.id) for i in self.inputs]
            base_name += f", inputs={input_names}"
        return base_name
    


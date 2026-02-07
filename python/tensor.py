import ctypes
from .compiler import Compiler
from .ops import Node, Operators
from typing import List, Optional, Tuple, Set
import numpy as np
class Tensor: 
    def __init__(self, name: str = None ,  node: Node= None, shape: Tuple[int, ...]=None, requires_grad: bool = False, data=None) -> None: 
        if node:
            self.node = node
            self.shape = node.shape
        else:
            if shape is None: 
                raise ValueError("Shape is required!!")
            self.node = Node(Operators.INPUT, name=name, shape=shape, requires_grad=requires_grad)
            self.shape = shape
            
        if data is not None:
             self.node.data = data

    @property
    def data(self):
        return self.node.data
    
    @data.setter
    def data(self, value):
        self.node.data = value

    @property
    def requires_grad(self) -> bool:
        return self.node.requires_grad
    
    def _prepare_ptr_array(self, arrays: List[Optional[np.ndarray]]):
        float_ptr = ctypes.POINTER(ctypes.c_float)
        ptr_array = (float_ptr * len(arrays))()
        kept_references = []

        for i, arr in enumerate(arrays):
            if arr is None:
                # ✅ Better error message
                raise RuntimeError(
                    f"Cannot create pointer array: element {i} is None. "
                    f"This usually means a node's .data wasn't allocated. "
                    f"Call forward() before backward()!"
                )
            
            # Ensure float32 and contiguous
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Element {i} is not a numpy array: {type(arr)}")
            
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            kept_references.append(arr)
            ptr_array[i] = arr.ctypes.data_as(float_ptr)
        
        return ptr_array, kept_references

    def forward(self) -> None:
        nodes_to_compute = []
        visited = set()
        def _topo(n):
            if n.id not in visited:
                visited.add(n.id)
                [_topo(inp) for inp in n.inputs]
                nodes_to_compute.append(n)
        _topo(self.node)

        for node in nodes_to_compute:
            if node.grad is not None:
                node.grad.fill(0.0)
            if node.op_type == Operators.INPUT:
                if node.data is None:
                    raise RuntimeError(f"INPUT node {node.name} has no data!")
                continue

            # HANDLE RESHAPE in Python
            if node.op_type == Operators.RESHAPE:
                inp_data = node.inputs[0].data
                if inp_data is None:
                    raise RuntimeError(f"Input to RESHAPE has no data!")
                node.data = inp_data.reshape(node.shape)
                continue
            
            if node.data is None:
                node.data = np.zeros(node.shape, dtype=np.float32)
            
            # Allocate output buffer
            if node.data is None:
                node.data = np.zeros(node.shape, dtype=np.float32)
            
            # Verify all inputs have data
            for inp in node.inputs:
                if inp.data is None:
                    raise RuntimeError(f"Input node {inp.name} has no data before computing {node.op_type.name}")
            
            compiler = Compiler()
            lib = compiler.jit_compile(node)
            
            if node.op_type in (Operators.MATMUL, Operators.LAYERNORM, Operators.SOFTMAX_CROSS_ENTROPY, Operators.GELU, Operators.MHA):
                all_data = [node.data] + [inp.data for inp in node.inputs]
                vars_ptr, _keep = self._prepare_ptr_array(all_data)
                lib.compute(vars_ptr, node.shape[0])
            elif node.op_type == Operators.FFN:
                X = node.inputs[0].data
                W1 = node.inputs[1].data
                b1 = node.inputs[2].data
                W2 = node.inputs[3].data
                b2 = node.inputs[4].data
                if not W2.flags['C_CONTIGUOUS'] or W2.shape[0] == node.attributes['hidden_dim']:
                    W2_T = np.ascontiguousarray(W2.T)
                else:
                    W2_T = W2
                all_data = [node.data, X, W1, b1, W2_T, b2]
                vars_ptr, _keep = self._prepare_ptr_array(all_data)
                lib.compute(vars_ptr, node.shape[0])
            else:
                chain_nodes = compiler.execution_order
                for n in chain_nodes:
                    if n.data is None:
                        n.data = np.zeros(n.shape, dtype=np.float32)
                vars_ptr, _keep = self._prepare_ptr_array([n.data for n in chain_nodes])
                total_elements = int(np.prod(node.shape))
                lib.compute(vars_ptr, total_elements)

    def backward(self) -> None: 
        if self.node.grad is None: self.node.grad = np.ones(self.shape, dtype=np.float32)

        # Full recursive backpropagation
        nodes = []
        visited = set()
        def _topo(n):
            if n.id not in visited:
                visited.add(n.id); [ _topo(inp) for inp in n.inputs ]
                nodes.append(n)
        _topo(self.node)

        for node in reversed(nodes):
            if not node.requires_grad or node.op_type == Operators.INPUT: continue
            
            for inp in node.inputs:
                if inp.grad is None: inp.grad = np.zeros(inp.shape, dtype=np.float32)
            
            if hasattr(node, 'backward_fn') and node.backward_fn is not None:
                saved = getattr(node, 'saved_tensors', None)
                node.backward_fn(node.grad, saved)
                continue
            
            # HANDLE RESHAPE in Backward
            if node.op_type == Operators.RESHAPE:
                node.inputs[0].grad += node.grad.reshape(node.inputs[0].shape)
                continue

            compiler = Compiler()
            lib, func_name = compiler.jit_compile_backward(node)
            func = getattr(lib, func_name)
            if node.op_type == Operators.FFN:
                X = node.inputs[0]
                W1 = node.inputs[1]
                b1 = node.inputs[2]
                W2 = node.inputs[3]
                b2 = node.inputs[4]
                if not W2.grad.flags['C_CONTIGUOUS']:
                    W2.grad = np.ascontiguousarray(W2.grad)
                g_p, _k1 = self._prepare_ptr_array([
                    X.grad, W1.grad, b1.grad, W2.grad, b2.grad
                ])
                v_p, _k2 = self._prepare_ptr_array([
                    X.data, W1.data, b1.data, 
                    np.ascontiguousarray(W2.data.T), b2.data
                ])
                grad_out_ptr = node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                func(g_p, v_p, node.shape[0], grad_out_ptr)
            elif func_name == "loss_backward":
                g_p, _k1 = self._prepare_ptr_array([node.inputs[0].grad])
                v_p, _k2 = self._prepare_ptr_array([node.inputs[0].data, node.inputs[1].data])
                # Pass batch size (dim 0) explicitly
                B = node.inputs[0].shape[0] if len(node.inputs[0].shape) == 2 else node.inputs[0].shape[0] * node.inputs[0].shape[1]
                func(g_p, v_p, B, node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            # ... [Rest of cases same] ...
            elif node.op_type == Operators.MATMUL:
                 g_p, _k1 = self._prepare_ptr_array([node.inputs[0].grad, node.inputs[1].grad])
                 v_p, _k2 = self._prepare_ptr_array([node.inputs[0].data, node.inputs[1].data])
                 func(g_p, v_p, node.shape[0], node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            

            elif node.op_type == Operators.MHA:
                # CRITICAL FIX: Handle self-attention where Q=K=V=same node.
                # If inputs share the same node, the kernel writes gQ, gK, gV
                # to the same pointer → data race under OpenMP.
                # Solution: allocate separate temp gradient buffers, then sum.
                
                q_node, k_node, v_node = node.inputs[0], node.inputs[1], node.inputs[2]
                
                # Check if any inputs are aliased (self-attention)
                ids = [q_node.id, k_node.id, v_node.id]
                has_aliasing = len(set(ids)) < 3
                
                if has_aliasing:
                    # Allocate separate gradient buffers
                    gQ_tmp = np.zeros(q_node.shape, dtype=np.float32)
                    gK_tmp = np.zeros(k_node.shape, dtype=np.float32)
                    gV_tmp = np.zeros(v_node.shape, dtype=np.float32)
                    
                    g_p, _k1 = self._prepare_ptr_array([gQ_tmp, gK_tmp, gV_tmp])
                    v_p, _k2 = self._prepare_ptr_array([q_node.data, k_node.data, v_node.data])
                    grad_out_ptr = node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    func(g_p, v_p, node.shape[0], grad_out_ptr)
                    
                    # Now accumulate into the actual gradient buffers (no race)
                    q_node.grad += gQ_tmp
                    k_node.grad += gK_tmp
                    v_node.grad += gV_tmp
                else:
                    # No aliasing — direct dispatch is safe
                    g_p, _k1 = self._prepare_ptr_array([q_node.grad, k_node.grad, v_node.grad])
                    v_p, _k2 = self._prepare_ptr_array([q_node.data, k_node.data, v_node.data])
                    grad_out_ptr = node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    func(g_p, v_p, node.shape[0], grad_out_ptr)


            elif node.op_type == Operators.LAYERNORM:
                x_inp, gamma_inp, beta_inp = node.inputs[0], node.inputs[1], node.inputs[2]
                if len(node.shape) == 2:
                    B, F = node.shape
                else:
                    B = node.shape[0] * node.shape[1]
                    F = node.shape[2]
                func(
                    x_inp.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    gamma_inp.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    beta_inp.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    x_inp.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    gamma_inp.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(B),
                    ctypes.c_int(F)
                )
            elif node.op_type == Operators.GELU:
                g_p, _k1 = self._prepare_ptr_array([node.inputs[0].grad])
                v_p, _k2 = self._prepare_ptr_array([node.inputs[0].data])
                func(g_p, v_p, node.shape[0], node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            else:
                chain = compiler.execution_order
                g_p, _k1 = self._prepare_ptr_array([n.grad for n in chain])
                v_p, _k2 = self._prepare_ptr_array([n.data for n in chain])
                func(g_p, v_p, node.shape[0], node.grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def zero_grad(self) -> None:
        if self.node.grad is not None:
             self.node.grad.fill(0.0)
    # operators
    def __add__(self, other: 'Tensor') -> 'Tensor':
        if self.shape != other.shape: 
            raise ValueError(f"Shape Mismatch: {self.shape} vs. {other.shape}")
        
        needs_grad = self.requires_grad or other.requires_grad
        new_node = Node(Operators.ADD, inputs=[self.node, other.node], shape=self.shape, requires_grad=needs_grad)

        if new_node.requires_grad: 
            # ADD backward is trivial — keep in Python (avoids JIT overhead)
            def backward_fn(grad_output, saved_tensor):
                if self.node.requires_grad:
                    if self.node.grad is None:
                        self.node.grad = np.zeros(self.node.shape, dtype=np.float32)
                    self.node.grad += grad_output 
                if other.node.requires_grad:
                    if other.node.grad is None:
                        other.node.grad = np.zeros(other.node.shape, dtype=np.float32)
                    other.node.grad += grad_output 
            
            new_node.backward_fn = backward_fn
        return Tensor(node = new_node)
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        if self.shape != other.shape: 
            raise ValueError(f"Shape Mismatch: {self.shape} vs. {other.shape}") 
        needs_grad = self.requires_grad or other.requires_grad
        new_node = Node(Operators.SUB, inputs=[self.node, other.node], shape=self.shape, requires_grad=needs_grad)
        if new_node.requires_grad: 
            def backward_fn(grad_output, saved_tensor):
                if self.node.requires_grad:
                    if self.node.grad is None:
                        self.node.grad = np.zeros(self.node.shape, dtype=np.float32)
                    self.node.grad += grad_output
                if other.node.requires_grad:
                    if other.node.grad is None:
                        other.node.grad = np.zeros(other.node.shape, dtype=np.float32)
                    other.node.grad -= grad_output
            
            new_node.backward_fn = backward_fn
        
        return Tensor(node = new_node)
    
    def __mul__(self, other: 'Tensor') -> 'Tensor': 
        if self.shape != other.shape: 
            raise ValueError(f"Shape Mismatch: {self.shape} vs. {other.shape}")
        needs_grad = self.requires_grad or other.requires_grad
        new_node = Node(Operators.MUL, inputs=[self.node, other.node], shape=self.shape, requires_grad=needs_grad)
        if new_node.requires_grad: 
            new_node.saved_tensors = [self, other]
            def backward_fn(grad_output, saved_tensors):
                a, b = saved_tensors
                if a.node.requires_grad:
                    if a.node.grad is None:
                        a.node.grad = np.zeros(a.node.shape, dtype=np.float32)
                    a.node.grad += grad_output * b.data
                if b.node.requires_grad:
                    if b.node.grad is None:
                        b.node.grad = np.zeros(b.node.shape, dtype=np.float32)
                    b.node.grad += grad_output * a.data
            
            new_node.backward_fn = backward_fn

        return Tensor(node=new_node)
    
    def __div__(self, other: 'Tensor') -> 'Tensor':
        if self.shape != other.shape: 
            raise ValueError(f"Shape Mismatch: {self.shape} vs. {other.shape}") 
        
        needs_grad = self.requires_grad or other.requires_grad
        new_node = Node(Operators.DIV, inputs=[self.node, other.node], shape=self.shape, requires_grad=needs_grad)
        if new_node.requires_grad:
            new_node.saved_tensors = [self, other]
            def backward_fn(grad_output, saved_tensors):
                a, b = saved_tensors
                if a.node.requires_grad:
                    if a.node.grad is None:
                        a.node.grad = np.zeros(a.node.shape, dtype=np.float32)
                    a.node.grad += grad_output / b.data
                if b.node.requires_grad:
                    if b.node.grad is None:
                        b.node.grad = np.zeros(b.node.shape, dtype=np.float32)
                    b.node.grad -= grad_output * a.data / (b.data ** 2)
            new_node.backward_fn = backward_fn
        return Tensor(node=new_node)
    
    __truediv__ = __div__
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if self.shape[1] != other.shape[0]: 
            raise ValueError(f"MatMul Shape Mismatch: {self.shape[1]} vs. {other.shape[0]}")
        needs_grad = self.requires_grad or other.requires_grad

        result_shape = (self.shape[0], other.shape[1])
        new_node = Node(Operators.MATMUL, inputs=[self.node, other.node], shape=result_shape, requires_grad=needs_grad)

        # FIX: backward_fn = None → falls through to JIT matmul backward
        # (The old Python np.matmul backward also worked, but JIT is faster)

        return Tensor(node=new_node)
    
    def transpose(self, perm: Optional[List[int]] = None) -> 'Tensor':
        if len(self.shape) != 2:
            raise ValueError("Transpose currently only supports 2D matrices in this version.")
        
        new_shape = (self.shape[1], self.shape[0])
        new_node = Node(Operators.TRANSPOSE, inputs=[self.node], shape=new_shape, requires_grad=self.requires_grad)
        # backward_fn stays None — handled in backward() or not needed
        return Tensor(node=new_node)
    
    def relu(self) -> 'Tensor':
        new_shape = self.shape
        new_node = Node(Operators.RELU, inputs=[self.node], shape=new_shape, requires_grad=self.requires_grad)
        # backward_fn = None → JIT backward handles ReLU
        return Tensor(node=new_node)
    
    def layernorm(self, weight: Optional['Tensor'] = None, bias: Optional['Tensor'] = None, eps: float = 1e-5) -> 'Tensor':
        new_node = Node(Operators.LAYERNORM, 
            inputs=[self.node, weight.node, bias.node],
            attributes={'eps': eps},
            shape=self.shape,
            requires_grad=True)
        
        # FIX: backward_fn = None → JIT compile_LayerNorm_Backend_SIMD handles this
        # Old code had `def backward_fn(...): pass` which silently ate all gradients
        
        return Tensor(node=new_node)
    

    def softmax_cross_entropy(self, labels: 'Tensor') -> 'Tensor':
        new_node = Node(Operators.SOFTMAX_CROSS_ENTROPY, inputs=[self.node, labels.node], shape=(self.shape[0],), requires_grad=True)
        
        # FIX: backward_fn = None → JIT compile_fused_loss_backward handles this
        # This is where the gradient chain STARTS — if this is `pass`, nothing trains
        
        return Tensor(node=new_node)

    def mha(self, num_heads: int, key: 'Tensor', value: 'Tensor') -> 'Tensor':
        new_node = Node(Operators.MHA, 
                        inputs=[self.node, key.node, value.node], 
                        attributes={'num_heads': num_heads},
                        shape=self.shape)
        if self.requires_grad or key.requires_grad or value.requires_grad:
            new_node.requires_grad = True
        
        # FIX: backward_fn = None → JIT compile_mha_backward handles this
        
        return Tensor(node=new_node)
    
    def gelu(self) -> 'Tensor': 
        new_shape = self.shape
        new_node = Node(Operators.GELU, inputs=[self.node], shape=new_shape, requires_grad=self.requires_grad)
        # FIX: backward_fn = None → JIT compile_GELU_Backward_SIMD handles this
        return Tensor(node=new_node)
    
    def ffn(self, W1: 'Tensor', b1: 'Tensor', W2: 'Tensor', b2: 'Tensor', hidden_dim: int) -> 'Tensor': 
        new_node = Node(
            Operators.FFN, 
            inputs=[self.node, W1.node, b1.node, W2.node, b2.node], 
            attributes={'hidden_dim': hidden_dim}, 
            shape=self.shape, 
            requires_grad=self.requires_grad or W1.requires_grad or W2.requires_grad
        )
        
        # FIX: backward_fn = None → JIT compile_FFN_backward_SIMD handles this
        
        return Tensor(node=new_node)

    def reshape(self, new_shape: List[int]) -> 'Tensor':
        new_node = Node(Operators.RESHAPE, inputs=[self.node], shape=tuple(new_shape), requires_grad=self.requires_grad)
        # backward_fn = None → handled by explicit RESHAPE check in backward()
        return Tensor(node=new_node)
    
    def flatten(self) -> 'Tensor':
        total = np.prod(self.shape)
        dim1 = total // self.shape[0]
        return self.reshape([self.shape[0], dim1])

    def sigmoid(self) -> 'Tensor':
        new_node = Node(Operators.SIGMOID, inputs=[self.node])
        return Tensor(node=new_node)
    
    def tanh(self) -> 'Tensor':
        new_node = Node(Operators.TANH, inputs=[self.node])
        return Tensor(node=new_node)
    
    def softmax(self) -> 'Tensor':
        new_node = Node(Operators.SOFTMAX, inputs=[self.node])
        return Tensor(node=new_node)
    
    def conv2d(self, filters: int, kernel_size: int, stride: int = 1, padding: str = 'valid') -> 'Tensor': 
        new_node = Node(Operators.CONV2D, inputs=[self.node])
        return Tensor(node=new_node)
    
    def maxpool2d(self, pool_size: int, stride: int = 2, padding: str = 'valid') -> 'Tensor': 
        new_node = Node(Operators.MAXPOOL2D, inputs=[self.node])
        return Tensor(node=new_node)

    
    def dense(self, units: int, activation: Optional[str] = None) -> 'Tensor':
        new_node = Node(Operators.DENSE, inputs=[self.node])
        return Tensor(node=new_node)
    def avgpool2d(self, pool_size: int, stride: int = 2, padding: str= 'valid') -> 'Tensor':
        new_node = Node(Operators.AVGPOOL2D, inputs=[self.node])
        return Tensor(node=new_node)

    def dropout(self, rate: float) -> 'Tensor':
        new_node = Node(Operators.DROPOUT, inputs=[self.node])
        return Tensor(node=new_node)

    def batchnorm(self) -> 'Tensor':
        new_node = Node(Operators.BATCHNORM, inputs=[self.node])
        return Tensor(node=new_node)


    def __concat__(self, other: 'Tensor', axis: int = 0) -> 'Tensor': 
        new_node = Node(Operators.CONCAT, inputs=[self.node, other.node], attributes={'axis': axis})
        return Tensor(node=new_node)
    

    
    def split(self, num_splits: int, axis: int = 0) -> List['Tensor']:
        new_node = Node(Operators.SPLIT, inputs=[self.node], attributes={'num_splits': num_splits, 'axis': axis})
        return [Tensor(node=new_node) for _ in range(num_splits)]
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        new_node = Node(Operators.SUM, inputs=[self.node])
        return Tensor(node=new_node)
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        new_node = Node(Operators.MEAN, inputs=[self.node])
        return Tensor(node=new_node)
    
    def variance(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        new_node = Node(Operators.VARIANCE, inputs=[self.node])
        return Tensor(node=new_node)
    
    def exp(self) -> 'Tensor':
        new_node = Node(Operators.EXP, inputs=[self.node])
        return Tensor(node=new_node)
    
    def argmax(self, axis: Optional[int] = None) -> 'Tensor':
        new_node = Node(Operators.ARGMAX, inputs=[self.node])
        return Tensor(node=new_node)
    
    def argmin(self, axis: Optional[int] = None) -> 'Tensor':
        new_node = Node(Operators.ARGMIN, inputs=[self.node])
        return Tensor(node=new_node)
    
    def log(self) -> 'Tensor':
        new_node = Node(Operators.LOG, inputs=[self.node])
        return Tensor(node=new_node)
    
    def sqrt(self) -> 'Tensor':
        new_node = Node(Operators.SQRT, inputs=[self.node])
        return Tensor(node=new_node)
    
    def power(self, exponent: float) -> 'Tensor':
        new_node = Node(Operators.POWER, inputs=[self.node])
        return Tensor(node=new_node)
    
    def clip(self, min_value: float, max_value: float) -> 'Tensor':
        new_node = Node(Operators.CLIP, inputs=[self.node])
        return Tensor(node=new_node)
    
    def slice(self, begin: List[int], size: List[int]) -> 'Tensor':
        new_node = Node(Operators.SLICE, inputs=[self.node])
        return Tensor(node=new_node)
    
    def gather(self, indices: 'Tensor', axis: int = 0) -> 'Tensor':
        new_node = Node(Operators.GATHER, inputs=[self.node, indices.node])
        return Tensor(node=new_node)
    
    def scatter(self, indices: 'Tensor', updates: 'Tensor', axis: int = 0) -> 'Tensor':
        new_node = Node(Operators.SCATTER, inputs=[self.node, indices.node, updates.node])
        return Tensor(node=new_node)
    
    def broadcast(self, shape: List[int]) -> 'Tensor':
        new_node = Node(Operators.BROADCAST, inputs=[self.node])
        return Tensor(node=new_node)
    
    def pad(self, paddings: List[List[int]], mode: str = 'constant', constant_values: float = 0) -> 'Tensor':
        new_node = Node(Operators.PAD, inputs=[self.node])
        return Tensor(node=new_node)
    
    def lstm(self, units: int, return_sequences: bool = False) -> 'Tensor':
        new_node = Node(Operators.LSTM, inputs=[self.node])
        return Tensor(node=new_node)
    
    def gru(self, units: int, return_sequences: bool = False) -> 'Tensor': 
        new_node = Node(Operators.GRU, inputs=[self.node])
        return Tensor(node=new_node)
    
    def normalize(self, axis: Optional[int] = None, epsilon: float = 1e-8) -> 'Tensor':
        new_node = Node(Operators.NORMALIZE, inputs=[self.node])
        return Tensor(node=new_node)
    
    def __repr__(self) -> str: 
        return f"Tensor(node={self.node}) operation={self.node.op_type.name}"

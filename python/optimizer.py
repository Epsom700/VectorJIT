import numpy as np
from typing import List
from .tensor import Tensor

class AdamOptimizer:
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # State buffers: m (1st moment) and v (2nd moment)
        self.m = {id(p.node): np.zeros_like(p.data) for p in self.parameters}
        self.v = {id(p.node): np.zeros_like(p.data) for p in self.parameters}

    def zero_grad(self):
        """Reset gradients to zero before the next backward pass."""
        for p in self.parameters:
            if p.node.grad is not None:
                # We use in-place fill to maintain the pointer address for C++
                p.node.grad.fill(0.0)

    def step(self):
        """Perform the Adam update step."""
        self.t += 1
        for p in self.parameters:
            node = p.node
            if node.grad is None:
                continue
            
            node_id = id(node)
            grad = node.grad
            
            # Update biased first moment estimate
            self.m[node_id] = self.beta1 * self.m[node_id] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[node_id] = self.beta2 * self.v[node_id] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[node_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[node_id] / (1 - self.beta2 ** self.t)
            
            # Update parameters: w = w - lr * m_hat / (sqrt(v_hat) + eps)
            node.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
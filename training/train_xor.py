import numpy as np
import os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.optimizer import AdamOptimizer
from python.tensor import Tensor
from python.compiler import Compiler

# 1. Setup Data (XOR: 4 samples, 2 features, 2 classes)
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_data = np.array([0, 1, 1, 0], dtype=np.float32) # Labels as float for C++ compatibility

# 2. Initialize Parameters (A tiny 2-layer MLP)
# W1: (2, 4), W2: (4, 2)
W1 = Tensor(name="W1", shape=(2, 4), requires_grad=True, 
            data=np.random.randn(2, 4).astype(np.float32) * 0.1)
W2 = Tensor(name="W2", shape=(4, 2), requires_grad=True, 
            data=np.random.randn(4, 2).astype(np.float32) * 0.1)

optimizer = AdamOptimizer([W1, W2], lr=0.1)

print("Starting Training...")

for epoch in range(100):
    # Prepare Inputs
    X = Tensor(name="X", shape=(4, 2), data=X_data)
    Y = Tensor(name="Y", shape=(4,), data=Y_data)
    
    # Build graph
    hidden = (X @ W1).gelu()
    logits = hidden @ W2
    loss_tensor = logits.softmax_cross_entropy(Y)
    
    # ✅ Execute entire graph in one forward pass
    loss_tensor.forward()
    current_loss = np.mean(loss_tensor.data)
    
    # Backward
    optimizer.zero_grad()


    loss_tensor.backward()
    optimizer.clip_gradients(max_norm=1.0)
    # Update
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {current_loss:.4f}")

print("Training Complete!")
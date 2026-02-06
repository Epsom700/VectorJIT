import numpy as np
import os, sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.optimizer import AdamOptimizer
from python.tensor import Tensor
from python.compiler import Compiler
from python.clear_cache import clear_cache

clear_cache()

def test_transformer_block():
    # ... [Setup same as before] ...
    batch_size = 4
    seq_len = 16
    model_dim = 512
    num_heads = 8
    hidden_dim = 2048 
    lr = 1e-4

    print(f"🚀 Initializing Transformer Block (MHA + FFN)...")
    # ... [Init tensors same as before] ...
    x_data = np.random.randn(batch_size, seq_len, model_dim).astype(np.float32) * 0.02
    x = Tensor(name="input_x", shape=x_data.shape, data=x_data, requires_grad=True)
    
    # ... [Weights init same] ...
    w1_data = (np.random.randn(model_dim, hidden_dim) * np.sqrt(2./model_dim)).astype(np.float32)
    w2_data = (np.random.randn(hidden_dim, model_dim) * np.sqrt(2./hidden_dim)).astype(np.float32)
    b1_data = np.zeros(hidden_dim, dtype=np.float32)
    b2_data = np.zeros(model_dim, dtype=np.float32)
    W1 = Tensor(name="W1", shape=w1_data.shape, data=w1_data, requires_grad=True)
    b1 = Tensor(name="b1", shape=b1_data.shape, data=b1_data, requires_grad=True)
    W2 = Tensor(name="W2", shape=w2_data.shape, data=w2_data, requires_grad=True)
    b2 = Tensor(name="b2", shape=b2_data.shape, data=b2_data, requires_grad=True)

    gamma1 = Tensor(name="gamma1", shape=(model_dim,), data=np.ones(model_dim, dtype=np.float32), requires_grad=True)
    beta1 = Tensor(name="beta1", shape=(model_dim,), data=np.zeros(model_dim, dtype=np.float32), requires_grad=True)
    gamma2 = Tensor(name="gamma2", shape=(model_dim,), data=np.ones(model_dim, dtype=np.float32), requires_grad=True)
    beta2 = Tensor(name="beta2", shape=(model_dim,), data=np.zeros(model_dim, dtype=np.float32), requires_grad=True)

    target_data = np.random.randint(0, model_dim, (batch_size * seq_len,)).astype(np.float32)
    targets = Tensor(name="targets", shape=target_data.shape, data=target_data)

    all_params = [W1, b1, W2, b2, gamma1, beta1, gamma2, beta2]
    optimizer = AdamOptimizer(all_params, lr=lr)

    print(f"🔥 Starting training loop with MHA + FFN Transformer Block...")
    
    for epoch in range(10):
        start_time = time.time()
        
        # --- FORWARD PASS ---
        mha_out = x.mha(num_heads=num_heads, key=x, value=x)
        residual1 = x + mha_out
        normed1 = residual1.layernorm(weight=gamma1, bias=beta1)
        
        ffn_out = normed1.ffn(W1, b1, W2, b2, hidden_dim=hidden_dim)
        residual2 = normed1 + ffn_out
        normed2 = residual2.layernorm(weight=gamma2, bias=beta2)
        
        # FIX: Flatten 3D (B, S, E) -> 2D (B*S, E) for CrossEntropy
        # normed2 is (4, 16, 512). Reshape to (64, 512)
        logits = normed2.reshape([batch_size * seq_len, model_dim])
        
        loss = logits.softmax_cross_entropy(targets)
        
        loss.forward()
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.clip_gradients(max_norm=1.0)
        optimizer.step()
        
        end_time = time.time()
        avg_loss = np.mean(loss.data)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {(end_time - start_time)*1000:.2f}ms")

if __name__ == "__main__":
    test_transformer_block()
import numpy as np
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.optimizer import AdamOptimizer
from python.tensor import Tensor
from python.compiler import Compiler
from python.clear_cache import clear_cache

import os
os.environ["OMP_NUM_THREADS"] = "1"

clear_cache()
np.random.seed(42)

def test_transformer_small():
    # =====================================================================
    # SMALL CONFIG — designed to converge, not to scale
    # =====================================================================
    batch_size = 4
    seq_len = 8
    model_dim = 64       # was 512
    num_heads = 4         # was 8  → head_dim = 16
    hidden_dim = 128      # was 2048
    num_classes = 32      # was 512 (= model_dim)
    lr = 1e-3
    num_epochs = 100

    print(f"Config: B={batch_size}, S={seq_len}, E={model_dim}, H={hidden_dim}, heads={num_heads}, classes={num_classes}")
    total_tokens = batch_size * seq_len  # 32

    # --- Input ---
    x_data = np.random.randn(batch_size, seq_len, model_dim).astype(np.float32) * 0.02
    x = Tensor(name="input_x", shape=x_data.shape, data=x_data, requires_grad=True)

    # --- FFN weights (smaller init for stability) ---
    w1_data = (np.random.randn(model_dim, hidden_dim) * np.sqrt(2.0 / model_dim)).astype(np.float32)
    w2_data = (np.random.randn(hidden_dim, model_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
    b1_data = np.zeros(hidden_dim, dtype=np.float32)
    b2_data = np.zeros(model_dim, dtype=np.float32)
    W1 = Tensor(name="W1", shape=w1_data.shape, data=w1_data, requires_grad=True)
    b1 = Tensor(name="b1", shape=b1_data.shape, data=b1_data, requires_grad=True)
    W2 = Tensor(name="W2", shape=w2_data.shape, data=w2_data, requires_grad=True)
    b2 = Tensor(name="b2", shape=b2_data.shape, data=b2_data, requires_grad=True)

    # --- LayerNorm params ---
    gamma1 = Tensor(name="gamma1", shape=(model_dim,), data=np.ones(model_dim, dtype=np.float32), requires_grad=True)
    beta1  = Tensor(name="beta1",  shape=(model_dim,), data=np.zeros(model_dim, dtype=np.float32), requires_grad=True)
    gamma2 = Tensor(name="gamma2", shape=(model_dim,), data=np.ones(model_dim, dtype=np.float32), requires_grad=True)
    beta2  = Tensor(name="beta2",  shape=(model_dim,), data=np.zeros(model_dim, dtype=np.float32), requires_grad=True)

    # --- Targets: random classes in [0, num_classes) ---
    # Note: num_classes must equal model_dim for the logits shape to work
    # (since the output of LN2 has dim model_dim, and CE expects (B*S, C))
    target_data = np.random.randint(0, model_dim, (total_tokens,)).astype(np.float32)
    targets = Tensor(name="targets", shape=target_data.shape, data=target_data)

    all_params = [W1, b1, W2, b2, gamma1, beta1, gamma2, beta2]
    optimizer = AdamOptimizer(all_params, lr=lr)

    param_count = sum(np.prod(p.node.data.shape) for p in all_params)
    print(f"Total parameters: {param_count:,}")
    print(f"Random baseline loss: ln({model_dim}) = {np.log(model_dim):.4f}")
    print(f"Target: loss < {np.log(model_dim)/2:.2f} (significant memorization)")
    print("-" * 60)

    for epoch in range(num_epochs):
        start = time.time()

        # --- Forward ---
        mha_out = x.mha(num_heads=num_heads, key=x, value=x)
        residual1 = x + mha_out
        normed1 = residual1.layernorm(weight=gamma1, bias=beta1)

        ffn_out = normed1.ffn(W1, b1, W2, b2, hidden_dim=hidden_dim)
        residual2 = normed1 + ffn_out
        normed2 = residual2.layernorm(weight=gamma2, bias=beta2)

        logits = normed2.reshape([total_tokens, model_dim])
        loss = logits.softmax_cross_entropy(targets)

        loss.forward()

        # --- Backward ---
        # Normalize seed gradient to per-sample mean
        loss.node.grad = np.ones(loss.shape, dtype=np.float32) / total_tokens
        optimizer.zero_grad()
        loss.backward()

        # --- Diagnostics ---
        param_norms = {}
        total_norm = 0.0
        for p in all_params:
            if p.node.grad is not None:
                pn = np.sqrt(np.sum(p.node.grad ** 2))
                param_norms[p.node.name] = pn
                total_norm += pn ** 2
        total_norm = np.sqrt(total_norm)

        # --- Update (NO clipping — let Adam handle it) ---
        optimizer.clip_gradients(1.0)
        optimizer.step()

        elapsed = (time.time() - start) * 1000
        avg_loss = np.mean(loss.data)

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Total Norm: {total_norm:.2f} | Time: {elapsed:.1f}ms")
            # Show which parameter dominates the gradient
            top = sorted(param_norms.items(), key=lambda x: -x[1])[:3]
            for name, norm in top:
                print(f"           {name:10s}: {norm:.2f}")

    print("-" * 60)
    final_loss = np.mean(loss.data)
    baseline = np.log(model_dim)
    if final_loss < baseline * 0.5:
        print(f"SUCCESS: Loss {final_loss:.4f} < {baseline*0.5:.4f} (half of random baseline)")
    elif final_loss < baseline * 0.9:
        print(f"PARTIAL: Loss decreased meaningfully ({final_loss:.4f}), pipeline works")
    else:
        print(f"STALLED: Loss {final_loss:.4f} barely moved from baseline {baseline:.4f}")
        print("  → Check grad norms above. If growing exponentially, there's a kernel bug.")
        print("  → If stable but small, increase lr.")

if __name__ == "__main__":
    test_transformer_small()
    os._exit(0)  # Prevent OpenMP cleanup segfault
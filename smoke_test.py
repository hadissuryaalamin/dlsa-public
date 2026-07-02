"""
Smoke test for CNNTransformer: verifies forward pass + training loop for both
vanilla (no GNN) and GNN (fully_connected) modes using synthetic data.
"""
import sys
import tempfile

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(False)

from models.CNNTransformer import CNNTransformer
from preprocess import preprocess_cumsum


def make_synthetic_residuals(T=100, N=20, missing_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, N)).astype(np.float32) * 0.01
    missing = rng.random((T, N)) < missing_frac
    data[missing] = 0.0
    return data


def _train_loop(model, windows, idxs_selected, data, lookback, n_epochs=2, batchsize=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T_win, N = windows.shape[0], windows.shape[1]
    use_gnn = getattr(model, "use_gnn", False)

    for epoch in range(n_epochs):
        for i in range(int(T_win / batchsize) + 1):
            batch_start = batchsize * i
            batch_end = min(batchsize * (i + 1), T_win)
            if batch_start >= batch_end:
                break

            idxs_batch = idxs_selected[batch_start:batch_end]  # (bs, N)

            if use_gnn:
                windows_batch = torch.tensor(windows[batch_start:batch_end])  # (bs, N, lookback)
                weights = model(windows_batch, valid_mask=idxs_batch)         # (bs, N)
            else:
                input_batch = windows[batch_start:batch_end][idxs_batch.numpy()]  # (n_valid, lookback)
                if input_batch.shape[0] == 0:
                    continue
                weights = torch.zeros((batch_end - batch_start, N))
                weights[idxs_batch] = model(torch.tensor(input_batch))

            assert not torch.isnan(weights).any(), "NaN in model output"
            assert not torch.isinf(weights).any(), "Inf in model output"

            abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True).clamp_min(1e-8)
            weights_norm = weights / abs_sum

            rets = torch.sum(
                weights_norm * torch.tensor(data[lookback + batch_start : lookback + batch_end]),
                dim=1,
            )
            loss = -torch.mean(rets) / (torch.std(rets) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"    epoch {epoch}: loss={loss.item():.4f}")


def test_no_gnn():
    print("\n[1/2] CNNTransformer  (use_gnn=False)")
    T, N, lookback = 100, 20, 10
    data = make_synthetic_residuals(T, N)
    windows, idxs_selected = preprocess_cumsum(data, lookback)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = CNNTransformer(
            logdir=tmpdir,
            random_seed=42,
            lookback=lookback,
            device="cpu",
            normalization_conv=True,
            filter_numbers=[1, 8],
            attention_heads=4,
            hidden_units_factor=2,
            dropout=0.25,
            filter_size=2,
            use_transformer=True,
            use_convolution=True,
            use_gnn=False,
        )
        model.train()
        _train_loop(model, windows, idxs_selected, data, lookback)

    print("  PASSED")


def test_gnn():
    print("\n[2/2] CNNTransformer + GNN  (use_gnn=True, gnn_type='fully_connected')")
    T, N, lookback = 100, 20, 10
    data = make_synthetic_residuals(T, N)
    windows, idxs_selected = preprocess_cumsum(data, lookback)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = CNNTransformer(
            logdir=tmpdir,
            random_seed=42,
            lookback=lookback,
            device="cpu",
            normalization_conv=True,
            filter_numbers=[1, 8],
            attention_heads=4,
            gnn_hidden_dim=8,
            hidden_units_factor=2,
            dropout=0.25,
            filter_size=2,
            use_transformer=True,
            use_convolution=True,
            use_gnn=True,
            gnn_type="fully_connected",
            gnn_layers=1,
        )
        model.train()
        _train_loop(model, windows, idxs_selected, data, lookback)

    print("  PASSED")


if __name__ == "__main__":
    test_no_gnn()
    test_gnn()
    print("\n=== All smoke tests passed ===")

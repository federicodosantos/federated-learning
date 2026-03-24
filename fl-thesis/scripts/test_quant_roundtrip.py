import logging
import sys

import torch
import numpy as np

# Adjust imports to match the project
from fl_thesis.quantization import quantize_parameters, dequantize_parameters

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_roundtrip():
    # 1. Create a dummy list of parameters imitating a small state_dict
    shapes = [
        (64, 3, 7, 7),
        (64,),
        (1000, 512),
        (1000,)
    ]

    parameters = []
    for s in shapes:
        # Create normal distributed float32 arrays
        p = np.random.randn(*s).astype(np.float32)
        parameters.append(p)

    print(f"Creating {len(parameters)} tensors for roundtrip test.")
    for i, p in enumerate(parameters):
        print(f"Tensor {i}: shape {p.shape}, elements {p.size}")

    print("\n--- Quantizing ---")
    try:
        quantized_bytes, params_list = quantize_parameters(parameters, bits=8)
    except Exception as e:
        print(f"Quantization failed: {e}")
        return

    print("\n--- Quantized Payloads ---")
    for i, (q, p) in enumerate(zip(quantized_bytes, parameters)):
        expected_len = p.size
        print(f"Tensor {i}: bytes length {len(q)}, expected {expected_len}, diff {len(q) - expected_len}")

    print("\n--- Dequantizing ---")
    try:
        dequantized = dequantize_parameters(quantized_bytes, params_list, shapes, bits=8)
    except Exception as e:
        print(f"Dequantization failed: {e}")
        return

    print("\n--- Validation ---")
    for i, (orig, deq) in enumerate(zip(parameters, dequantized)):
        assert orig.shape == deq.shape, f"Shape mismatch: {orig.shape} != {deq.shape}"
        # They won't be exactly equal due to quantization, but should be close
        diff = np.abs(orig - deq).mean()
        print(f"Tensor {i}: shape {deq.shape}, mean squared error: {diff:.4f}")

if __name__ == "__main__":
    test_roundtrip()

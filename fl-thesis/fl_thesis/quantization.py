import logging
from typing import List, Tuple

import numpy as np
import torch
from flwr.common import log
from torchao import quantization as torchao_quantization


def _compute_affine_params_per_channel(
    tensor: np.ndarray, bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scale and zero_point per-channel (per output channel / row).
    Robust terhadap 0-dim, 1-dim, dan array kosong.
    """
    # Jika scalar (0-dim), anggap 1 channel
    if tensor.ndim == 0:
        tensor_2d = tensor.reshape(1, -1)
    elif tensor.ndim == 1:
        tensor_2d = tensor.reshape(1, -1)
    else:
        tensor_2d = tensor.reshape(tensor.shape[0], -1)

    n_channels = tensor_2d.shape[0]
    scales = np.zeros(n_channels, dtype=np.float32)
    zero_points = np.zeros(n_channels, dtype=np.int32)

    for i in range(n_channels):
        ch = tensor_2d[i]
        # Jika channel kosong (bisa terjadi pada weird shapes), beri fallback
        if ch.size == 0:
            scales[i] = 1.0
            zero_points[i] = 0
            continue

        min_val = float(ch.min())
        max_val = float(ch.max())

        if max_val == min_val:
            scales[i] = 1.0
            zero_points[i] = int(np.round(-min_val))
        else:
            scales[i] = (max_val - min_val) / (2**bits - 1)
            # Karena output dtype dari TorchAO adalah int8 (signed, range -128 s.d 127), 
            # kita harus memastikan min_val berada di batas paling bawah yakni -128 (bukan 0)
            zero_points[i] = int(np.round(-128 - min_val / scales[i]))

    return scales, zero_points


def quantize_parameters(
    parameters: List[np.ndarray],
    bits: int = 8,
) -> Tuple[List[bytes], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Quantize a list of float32 numpy arrays using per-channel affine quantization.
    Ensure output bytes contain only the raw int8 data matching original shapes.
    """
    if bits != 8:
        raise NotImplementedError("Only 8-bit quantization is currently supported.")

    quantized_bytes = []
    params_list = []

    log(level=logging.INFO, msg="Start quantizing")
    for idx, param in enumerate(parameters):
        if param.dtype != np.float32:
            param = param.astype(np.float32)

        original_shape = param.shape

        # Hitung scale & zero_point per channel
        scales, zero_points = _compute_affine_params_per_channel(param, bits)

        # Reshape ke 2D untuk kuantisasi per channel
        if param.ndim == 0:
            param_2d = param.reshape(1, -1)
        elif param.ndim == 1:
            param_2d = param.reshape(1, -1)
        else:
            param_2d = param.reshape(param.shape[0], -1)

        n_channels = param_2d.shape[0]
        n_cols = param_2d.shape[1]
        q_2d = np.zeros((n_channels, n_cols), dtype=np.int8)

        for i in range(n_channels):
            ch_np = param_2d[i]
            ch_tensor = torch.from_numpy(ch_np.reshape(1, n_cols))

            # block_size=(1, n_kolom) → per-channel
            block_size = (1, n_cols)

            q_ch = torchao_quantization.quantize_affine(
                input=ch_tensor,
                block_size=block_size,
                scale=torch.tensor([scales[i]], dtype=torch.float32),
                zero_point=torch.tensor([zero_points[i]], dtype=torch.int32),
                output_dtype=torch.int8,
            )

            # Convert q_ch to plain numpy int8 array robustly:
            try:
                q_ch_np = (
                    q_ch.cpu().numpy()
                    if isinstance(q_ch, torch.Tensor)
                    else np.array(q_ch)
                )
            except Exception:
                # Fallback: attempt int_repr if available (some quantized types)
                if hasattr(q_ch, "int_repr"):
                    q_ch_np = q_ch.int_repr().cpu().numpy()
                elif hasattr(q_ch, "tensor"):
                    try:
                        q_ch_np = q_ch.tensor.cpu().numpy()
                    except Exception:
                        q_ch_np = np.array(q_ch)
                else:
                    q_ch_np = np.array(q_ch)

            assert q_ch_np.size == n_cols, f"Quantization size mismatch. Expected {n_cols}, got {q_ch_np.size}"

            q_2d[i] = q_ch_np.reshape(n_cols)

        # Reshape kembali ke original shape sebelum dijadikan bytes
        q_original = q_2d.reshape(original_shape)

        quantized_bytes.append(q_original.tobytes())
        params_list.append((scales, zero_points))

    return quantized_bytes, params_list


def dequantize_parameters(
    quantized_bytes: List[bytes],
    params_list: List[Tuple[np.ndarray, np.ndarray]],
    shapes: List[Tuple[int, ...]],
    bits: int = 8,
) -> List[np.ndarray]:
    """
    Dequantize bytes back to float32 numpy arrays using per-channel params.

    Robust handling:
    - If incoming byte buffer has 128 extra int8 elements (header/prefix), try trimming them.
    - Logs helpful messages when fallback happens.
    """
    if bits != 8:
        raise NotImplementedError("Only 8-bit dequantization is currently supported.")

    parameters = []

    for idx, (q_bytes, (scales, zero_points), shape) in enumerate(
        zip(quantized_bytes, params_list, shapes)
    ):
        try:
            # Convert bytes to numpy array of int8
            q_array = np.frombuffer(q_bytes, dtype=np.int8)
            expected_elems = None
            try:
                expected_elems = 1
                for d in shape:
                    expected_elems *= int(d)
            except Exception:
                expected_elems = None

            # Try initial reshape
            reshaped_ok = False
            try:
                q_array = q_array.reshape(shape)
                reshaped_ok = True
            except Exception as e:
                log(
                    level=logging.WARNING,
                    msg=(
                        f"[dequantize_parameters] tensor idx={idx}: cannot reshape array of size "
                        f"{q_array.size} into shape {shape} (expected elements={expected_elems}): {e}"
                    ),
                )
                raise RuntimeError(
                    f"tensor idx={idx}: reshape failed. Size {q_array.size} != expected {expected_elems}"
                )

            # At this point q_array has shape consistent with `shape`
            # Normalize to 2D per-channel view
            if q_array.ndim == 0:
                q_2d = q_array.reshape(1, 1)
            elif q_array.ndim == 1:
                q_2d = q_array.reshape(1, -1)
            else:
                q_2d = q_array.reshape(q_array.shape[0], -1)

            n_channels = q_2d.shape[0]
            n_cols = q_2d.shape[1]
            out_2d = np.zeros((n_channels, n_cols), dtype=np.float32)

            for i in range(n_channels):
                q_ch_np = q_2d[i].copy()
                q_ch_tensor = (
                    torch.from_numpy(q_ch_np).to(torch.int8).reshape(1, n_cols)
                )

                block_size = (1, n_cols)
                try:
                    ch_float = torchao_quantization.dequantize_affine(
                        input=q_ch_tensor,
                        block_size=block_size,
                        scale=torch.tensor([scales[i]], dtype=torch.float32),
                        zero_point=torch.tensor([zero_points[i]], dtype=torch.int32),
                        input_dtype=torch.int8,
                        output_dtype=torch.float32,
                    )
                    out_2d[i] = ch_float.cpu().numpy().reshape(n_cols)
                except Exception as e:
                    log(
                        level=logging.WARNING,
                        msg=(
                            f"[dequantize_parameters] tensor idx={idx} ch={i}: TorchAO dequantize_affine failed: {e}. Using manual dequantize."
                        ),
                    )
                    s = scales[i] if scales[i] != 0 else 1.0
                    zp = int(zero_points[i])
                    out_2d[i] = (q_ch_np.astype(np.float32) - zp) * s

            # Reshape back to original shape
            try:
                parameters.append(out_2d.reshape(shape).astype(np.float32))
            except Exception:
                log(
                    level=logging.WARNING,
                    msg=(
                        f"[dequantize_parameters] tensor idx={idx}: failed to reshape dequantized data back to {shape}. "
                        "Appending flattened array as fallback."
                    ),
                )
                parameters.append(out_2d.flatten().astype(np.float32))

        except Exception as e:
            log(
                level=logging.WARNING,
                msg=f"[dequantize_parameters] tensor idx={idx}: dequantization failed: {e}",
            )
            raise

    return parameters

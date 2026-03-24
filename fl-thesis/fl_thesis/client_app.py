import json
import logging
from typing import List

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, log, parameters_to_ndarrays
from flwr.common.parameter import bytes_to_ndarray
from torchvision.models.resnet import ResNet

from fl_thesis.quantization import dequantize_parameters, quantize_parameters
from fl_thesis.task import (
    DEVICE,
    get_weights,
    load_data,
    load_model,
    set_weights,
    test,
    train,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: ResNet,
        trainloader,
        valloader,
        local_epochs,
        quantization: str = "none",
        quantization_bits: int = 8,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = DEVICE
        self.quantization = quantization
        self.quantization_bits = quantization_bits
        self.tensor_shapes = [p.shape for p in net.state_dict().values()]
        self.net.to(self.device)

    def fit(self, parameters, config):
        log(level=logging.INFO, msg="[Client] Starting fit (training)...")
        log(level=logging.INFO, msg="Dequantizing models parameter from server...")
        
        ndarrays = self._maybe_dequantize_from_server(parameters, config)

        # Set weights & training lokal
        set_weights(self.net, ndarrays)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        # Ambil hasil training (float32)
        weights = get_weights(self.net)

        # --- Kuantisasi parameter lokal sebelum dikirim ke server ---
        if self.quantization != "none":
            log(
                level=logging.INFO,
                msg=f"[Client] Quantizing local weights ({self.quantization_bits}-bit per-channel)...",
            )
            try:
                quantized_bytes, params_list = quantize_parameters(
                    weights,
                    bits=self.quantization_bits,
                )

                scales_serialized = json.dumps([s.tolist() for s, _ in params_list])
                zero_points_serialized = json.dumps([zp.tolist() for _, zp in params_list])

                # Hitung ukuran payload untuk dikirim 
                original_size_kb = sum(w.nbytes for w in weights) / 1024
                quantized_size_kb = sum(len(q) for q in quantized_bytes) / 1024
                savings = (1.0 - quantized_size_kb / original_size_kb) * 100

                log(
                    level=logging.INFO,
                    msg=f"[Client] Upload: {original_size_kb:.2f} KB → {quantized_size_kb:.2f} KB ({savings:.1f}% reduction)",
                )

                # Ekstrak array kuantisasi int8 yang nantinya di-package oleh Flower via NP.Save (ada 128 byte headers)
                quantized_ndarrays = []
                for q, shape in zip(quantized_bytes, self.tensor_shapes):
                    arr = np.frombuffer(q, dtype=np.int8)
                    try:
                        quantized_ndarrays.append(arr.reshape(shape))
                    except Exception as e:
                        log(
                            level=logging.WARNING,
                            msg=f"[Client] Failed to reshape quantized tensor to shape={shape}. Error: {e}",
                        )
                        quantized_ndarrays.append(arr.reshape(-1))

                return (
                    quantized_ndarrays,
                    len(self.trainloader.dataset),
                    {
                        "train_loss": train_loss,
                        "quantization_scales": scales_serialized,
                        "quantization_zero_points": zero_points_serialized,
                    },
                )

            except Exception as e:
                log(
                    level=logging.ERROR,
                    msg=f"[Client] Quantization failed: {e}. Falling back to float32.",
                )

        return (
            weights,
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        log(level=logging.INFO, msg="[Client] Starting evaluate...")

        ndarrays = self._maybe_dequantize_from_server(parameters, config)
        set_weights(self.net, ndarrays)
        loss, accuracy = test(self.net, self.valloader, self.device)

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def _maybe_dequantize_from_server(self, parameters, config):
        """
        Convert incoming 'parameters' to a list of numpy ndarrays, handling both:
        - flwr.common.typing.Parameters (has .tensors)
        - plain Python list (e.g. list of bytes / ndarrays)
        """
        ndarrays = []

        if hasattr(parameters, "tensors"):
            ndarrays = parameters_to_ndarrays(parameters)
        elif isinstance(parameters, (list, tuple)):
            for idx, t in enumerate(parameters):
                if isinstance(t, np.ndarray):
                    ndarrays.append(t)
                elif isinstance(t, (bytes, bytearray)):
                    try:
                        ndarrays.append(bytes_to_ndarray(t))
                    except Exception:
                        ndarrays.append(np.frombuffer(t, dtype=np.float32))
                else:
                    try:
                        ndarrays.append(np.array(t))
                    except Exception as e:
                        raise RuntimeError(f"Unsupported parameter element at index {idx}: {type(t)} ({e})")
        else:
            raise TypeError(f"Unsupported parameters object type: {type(parameters)}")

        # Dekuantisasi parameter menjadi parameter riil (float32)
        if self.quantization != "none" and "quantization_scales" in config:
            scales_list = json.loads(config.get("quantization_scales", "[]"))
            zero_points_list = json.loads(config.get("quantization_zero_points", "[]"))
            
            params_list = [
                (np.array(s, dtype=np.float32), np.array(zp, dtype=np.int32))
                for s, zp in zip(scales_list, zero_points_list)
            ]

            q_bytes_list, shapes = [], []
            for arr in ndarrays:
                if isinstance(arr, np.ndarray) and arr.dtype == np.int8:
                    q_bytes_list.append(arr.tobytes())
                    shapes.append(arr.shape)
                elif isinstance(arr, (bytes, bytearray)):
                    q_bytes_list.append(bytes(arr))
                    shapes.append(tuple()) 
                else:
                    q_bytes_list.append(arr.tobytes())
                    shapes.append(arr.shape)

            ndarrays = dequantize_parameters(q_bytes_list, params_list, shapes, bits=8)

        return ndarrays


def client_fn(context: Context):
    net = load_model()
    trainloader, valloader = load_data()
    local_epochs = context.run_config["local-epochs"]
    quantization = context.run_config.get("quantization", "none")
    quantization_bits = context.run_config.get("quantization-bits", 8)

    log(
        level=logging.INFO,
        msg=f"[Client] Initialized: quantization={quantization}, bits={quantization_bits}",
    )

    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs,
        quantization=quantization,
        quantization_bits=quantization_bits,
    ).to_client()


app = ClientApp(client_fn)

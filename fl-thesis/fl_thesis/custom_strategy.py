import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_thesis.quantization import dequantize_parameters, quantize_parameters


class FedAvgWithCost(FedAvg):
    """FedAvg strategy dengan per-channel quantization dan communication cost tracking.

    Flow:
    1. configure_fit  : Kuantisasi global model (per-channel) → broadcast ke klien
                        Metadata (scales, zero_points) dikirim via FitIns.config
    2. aggregate_fit  : Terima upload klien (int8) → dekuantisasi → FedAvg agregasi
    3. aggregate_evaluate: Log akurasi per ronde
    """

    def __init__(
        self,
        *args,
        quantization: str = "none",
        quantization_bits: int = 8,
        shapes: Optional[List[Tuple[int, ...]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quantization = quantization
        self.quantization_bits = quantization_bits
        self.shapes = shapes or []
        self.last_aggregated_ndarrays = None

        # Communication cost tracking
        self.total_upload_cost_mb = 0.0
        self.total_download_cost_mb = 0.0
        self.communication_log = []

        log(
            level=logging.INFO,
            msg=f"FedAvgWithCost initialized: quantization={quantization} (per-channel), bits={quantization_bits}",
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Kuantisasi global model sebelum broadcast ke klien."""
        config = {}

        if self.quantization != "none":
            log(
                level=logging.INFO,
                msg=f"[Round {server_round}] Quantizing global model (per-channel) for broadcast...",
            )
            try:
                current_ndarrays = parameters_to_ndarrays(parameters)
                quantized_bytes, params_list = quantize_parameters(
                    current_ndarrays,
                    bits=self.quantization_bits,
                )

                # Serialisasi metadata
                scales_serialized = json.dumps([s.tolist() for s, _ in params_list])
                zero_points_serialized = json.dumps([zp.tolist() for _, zp in params_list])
                
                # Sinkronisasi shapes
                self.shapes = [arr.shape for arr in current_ndarrays]
                shapes_serialized = json.dumps([list(s) for s in self.shapes])

                # Konversi ke dict int8 ndarrays bagi parameter Flower
                quantized_ndarrays = [
                    np.frombuffer(q, dtype=np.int8).reshape(shape)
                    for q, shape in zip(quantized_bytes, self.shapes)
                ]
                parameters = ndarrays_to_parameters(quantized_ndarrays)

                # Simpan konfigurasi payload
                config.update(
                    {
                        "quantization_scales": scales_serialized,
                        "quantization_zero_points": zero_points_serialized,
                        "quantization_shapes": shapes_serialized,
                        "quantization": "asymmetric",
                        "quantization_bits": self.quantization_bits,
                    }
                )

                self._log_download_cost(server_round, quantized_bytes)
                log(
                    level=logging.INFO,
                    msg=f"[Round {server_round}] Global model quantized & metadata disertakan di config.",
                )

            except Exception as e:
                log(
                    level=logging.WARNING,
                    msg=f"[Round {server_round}] Gagal kuantisasi untuk broadcast: {e}. Menggunakan float32.",
                )
                if self.last_aggregated_ndarrays is not None:
                    float32_bytes = [arr.tobytes() for arr in self.last_aggregated_ndarrays]
                    self._log_download_cost(server_round, float32_bytes)
        else:
            if self.last_aggregated_ndarrays is not None:
                float32_bytes = [arr.tobytes() for arr in self.last_aggregated_ndarrays]
                self._log_download_cost(server_round, float32_bytes)

        # Meminta client configuration melalui strategy parent
        try:
            fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        except Exception as e:
            log(
                level=logging.WARNING,
                msg=f"[Round {server_round}] Error saat memanggil super().configure_fit: {e}",
            )
            return []

        # Injeksi metadata config kuantisasi ke tiap-tiap message
        if config:
            updated_fit_ins_list = []
            for client_proxy, fit_ins in fit_ins_list:
                merged_config = {**fit_ins.config, **config}
                updated_fit_ins = FitIns(parameters=parameters, config=merged_config)
                updated_fit_ins_list.append((client_proxy, updated_fit_ins))
            return updated_fit_ins_list

        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Dekuantisasi upload klien lalu agregasi dengan FedAvg."""
        
        # Log biaya upload berdasarkan kiriman payload aktual (int8) sebelum kita convert menjadi Float32
        self._log_upload_cost(server_round, results)

        if self.quantization != "none":
            log(
                level=logging.INFO,
                msg=f"[Round {server_round}] Dequantizing {len(results)} client uploads...",
            )
            results = self._dequantize_fit_results(results)

        # Agregasi FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.last_aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, metrics

    def _dequantize_fit_results(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Dekuantisasi hasil upload klien menggunakan per-channel metadata mereka."""
        dequantized_results = []

        for client, fit_res in results:
            metrics = fit_res.metrics or {}

            if "quantization_scales" not in metrics:
                dequantized_results.append((client, fit_res))
                continue

            try:
                scales_list = json.loads(metrics["quantization_scales"])
                zero_points_list = json.loads(metrics["quantization_zero_points"])

                params_list = [
                    (np.array(s, dtype=np.float32), np.array(zp, dtype=np.int32))
                    for s, zp in zip(scales_list, zero_points_list)
                ]

                # Decode representasi Flower Parameter → membersihkan 128-byte `.npy` header 
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                quantized_bytes = [arr.tobytes() for arr in ndarrays]

                dequantized_ndarrays = dequantize_parameters(
                    quantized_bytes,
                    params_list,
                    self.shapes,
                    bits=self.quantization_bits,
                )

                # Rekontruksi parameter float32 dan kembalikan metrik yang bersih
                dequantized_params = ndarrays_to_parameters(dequantized_ndarrays)
                new_metrics = {
                    k: v for k, v in metrics.items()
                    if k not in ["quantization_scales", "quantization_zero_points"]
                }
                
                new_fit_res = FitRes(
                    parameters=dequantized_params,
                    num_examples=fit_res.num_examples,
                    metrics=new_metrics,
                    status=fit_res.status,
                )
                dequantized_results.append((client, new_fit_res))

            except Exception as e:
                log(
                    level=logging.WARNING,
                    msg=f"[Server] Gagal dekuantisasi upload klien: {e}. Menggunakan parameter asli.",
                )
                dequantized_results.append((client, fit_res))

        return dequantized_results

    def _log_upload_cost(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """Log biaya komunikasi upload (klien → server)."""
        num_clients = len(results)
        upload_bytes_total = 0

        for _, fit_res in results:
            if fit_res.parameters is not None:
                upload_bytes_total += sum(len(t) for t in fit_res.parameters.tensors)

        upload_mb_per_client = upload_bytes_total / (1024 * 1024) / max(num_clients, 1)
        self.total_upload_cost_mb += upload_bytes_total / (1024 * 1024)

        savings_str = ""
        if self.quantization != "none" and self.last_aggregated_ndarrays is not None:
            float32_mb = sum(a.nbytes for a in self.last_aggregated_ndarrays) / (1024 * 1024)
            if float32_mb > 0:
                savings = (1.0 - upload_mb_per_client / float32_mb) * 100
                savings_str = f" ({savings:.1f}% hemat vs float32)"

        log(
            level=logging.INFO,
            msg=f"[Round {server_round}] Upload: {upload_mb_per_client:.4f} MB/client × {num_clients} clients{savings_str}",
        )

    def _log_download_cost(
        self,
        server_round: int,
        tensor_bytes: List[bytes],
    ) -> None:
        """Log biaya komunikasi download (server → klien)."""
        download_mb = sum(len(t) for t in tensor_bytes) / (1024 * 1024)
        self.total_download_cost_mb += download_mb

        savings_str = ""
        if self.quantization != "none" and self.last_aggregated_ndarrays is not None:
            float32_mb = sum(a.nbytes for a in self.last_aggregated_ndarrays) / (1024 * 1024)
            if float32_mb > 0:
                savings = (1.0 - download_mb / float32_mb) * 100
                savings_str = f" ({savings:.1f}% hemat vs float32)"

        log(
            level=logging.INFO,
            msg=f"[Round {server_round}] Download: {download_mb:.4f} MB/client{savings_str}",
        )
        log(
            level=logging.INFO,
            msg=f"[Round {server_round}] Cumulative — Upload: {self.total_upload_cost_mb:.4f} MB | Download: {self.total_download_cost_mb:.4f} MB",
        )

        self.communication_log.append(
            {
                "round": server_round,
                "upload_mb": self.total_upload_cost_mb,
                "download_mb": download_mb,
                "cumulative_mb": self.total_upload_cost_mb + self.total_download_cost_mb,
                "quantization": self.quantization,
            }
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Agregasi hasil evaluasi dan log akurasi per ronde."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if loss is not None and metrics:
            accuracy = metrics.get("accuracy", 0.0)
            log(
                level=logging.INFO,
                msg=f"[Round {server_round}] Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%",
            )

        return loss, metrics

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for server use
import matplotlib.pyplot as plt
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
        num_rounds: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quantization = quantization
        self.quantization_bits = quantization_bits
        self.shapes = shapes or []
        self.last_aggregated_ndarrays = None
        self.num_rounds = num_rounds

        # Communication cost tracking
        self.total_upload_cost_mb = 0.0
        self.total_download_cost_mb = 0.0
        self.communication_log = []

        # Accuracy history for plotting
        self.accuracy_history: List[Tuple[int, float]] = []  # (round, accuracy)

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
        # Bytes yang akan dikirim ke masing-masing klien (diisi di bawah)
        tensor_bytes_to_send: List[bytes] = []

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

                tensor_bytes_to_send = quantized_bytes
                log(
                    level=logging.INFO,
                    msg=f"[Round {server_round}] Global model quantized & metadata disertakan di config.",
                )

            except Exception as e:
                log(
                    level=logging.WARNING,
                    msg=f"[Round {server_round}] Gagal kuantisasi untuk broadcast: {e}. Menggunakan float32.",
                )
                # Fallback: gunakan parameter float32 dari hasil agregasi sebelumnya
                if self.last_aggregated_ndarrays is not None:
                    tensor_bytes_to_send = [arr.tobytes() for arr in self.last_aggregated_ndarrays]
                else:
                    tensor_bytes_to_send = [t for t in parameters.tensors]
        else:
            # Tidak ada kuantisasi — kirim parameter float32 apa adanya
            if self.last_aggregated_ndarrays is not None:
                tensor_bytes_to_send = [arr.tobytes() for arr in self.last_aggregated_ndarrays]
            else:
                # Round 1: gunakan initial parameters
                tensor_bytes_to_send = [t for t in parameters.tensors]

        # Meminta client configuration melalui strategy parent (menentukan jumlah klien yang disampling)
        try:
            fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        except Exception as e:
            log(
                level=logging.WARNING,
                msg=f"[Round {server_round}] Error saat memanggil super().configure_fit: {e}",
            )
            return []

        # Log biaya download SETELAH sampling agar kita tahu jumlah klien yang tepat
        num_sampled_clients = len(fit_ins_list)
        self._log_download_cost(server_round, tensor_bytes_to_send, num_sampled_clients)

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
        num_clients: int = 1,
    ) -> None:
        """Log biaya komunikasi download (server → setiap klien).

        Args:
            server_round: Ronde pelatihan saat ini.
            tensor_bytes: Bytes payload yang dikirim ke masing-masing klien.
            num_clients: Jumlah klien yang menerima broadcast pada ronde ini.
        """
        download_mb_per_client = sum(len(t) for t in tensor_bytes) / (1024 * 1024)
        # Total download = ukuran per klien × jumlah klien
        download_mb_total = download_mb_per_client * num_clients
        self.total_download_cost_mb += download_mb_total

        savings_str = ""
        if self.quantization != "none" and self.last_aggregated_ndarrays is not None:
            float32_mb = sum(a.nbytes for a in self.last_aggregated_ndarrays) / (1024 * 1024)
            if float32_mb > 0:
                savings = (1.0 - download_mb_per_client / float32_mb) * 100
                savings_str = f" ({savings:.1f}% hemat vs float32)"

        log(
            level=logging.INFO,
            msg=f"[Round {server_round}] Download: {download_mb_per_client:.4f} MB/client × {num_clients} clients"
                f" = {download_mb_total:.4f} MB total{savings_str}",
        )
        log(
            level=logging.INFO,
            msg=f"[Round {server_round}] Cumulative — Upload: {self.total_upload_cost_mb:.4f} MB | Download: {self.total_download_cost_mb:.4f} MB",
        )

        self.communication_log.append(
            {
                "round": server_round,
                "upload_mb": self.total_upload_cost_mb,
                "download_mb_per_client": download_mb_per_client,
                "download_mb_total": download_mb_total,
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
        """Agregasi hasil evaluasi dan log akurasi + biaya komunikasi per ronde."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if metrics:
            accuracy = metrics.get("accuracy", 0.0)
            # Btotal = upload + download (komunikasi dua arah)
            total_comm_cost_mb = self.total_upload_cost_mb + self.total_download_cost_mb
            log(
                level=logging.INFO,
                msg=f"[Round {server_round}] Accuracy: {accuracy:.2f}%"
                    f" | Comm Cost: {total_comm_cost_mb:.4f} MB",
            )

            # Sertakan biaya komunikasi total di metrics agar masuk ke History summary
            metrics["comm_cost_mb"] = round(total_comm_cost_mb, 4)

            # Catat akurasi per ronde untuk plotting
            self.accuracy_history.append((server_round, float(accuracy)))

            # Jika ini ronde terakhir, buat dan simpan grafik akurasi
            if server_round >= self.num_rounds:
                self._save_accuracy_plot()

        return loss, metrics

    def _save_accuracy_plot(self) -> None:
        """Buat dan simpan grafik akurasi global per ronde komunikasi."""
        if not self.accuracy_history:
            log(level=logging.WARNING, msg="[Plot] Tidak ada data akurasi untuk diplot.")
            return

        rounds = [r for r, _ in self.accuracy_history]
        accuracies = [acc for _, acc in self.accuracy_history]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(rounds, accuracies, marker="o", linewidth=2, markersize=6,
                color="#2196F3", label="Global Accuracy")

        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_title(
            f"Global Model Accuracy per Communication Round\n"
            f"(Quantization: {self.quantization})",
            fontsize=14,
        )
        ax.set_xticks(rounds)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()

        # Simpan grafik di direktori logs agar rapi
        output_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        quant_label = "quant" if self.quantization != "none" else "no-quant"
        output_path = os.path.join(output_dir, f"accuracy_plot_{quant_label}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        log(
            level=logging.INFO,
            msg=f"[Plot] Grafik akurasi disimpan di: {output_path}",
        )


import logging
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    log,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithCost(FedAvg):
    def __init__(self, *args, **kwargs):
        # Mewarisi semua inisialisasi dari FedAvg asli
        super().__init__(*args, **kwargs)

        # Variabel untuk menyimpan total biaya komunikasi
        self.total_communication_cost_mb = 0.0
        self.communication_log = []

        self.scales = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # 1. Panggil logika asli FedAvg untuk melakukan agregasi bobot
        #    Ini mengembalikan 'aggregated_parameters' (model global baru)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # 2. Sisipkan perhitungan Biaya Komunikasi di sini
        if aggregated_parameters is not None:
            # A. Hitung Ukuran Model (Model Size)
            # Flower menyimpan parameter sebagai list of byte arrays (tensors)
            # Kita hitung total bytes dari semua layer
            model_size_bytes = sum([len(t) for t in aggregated_parameters.tensors])
            model_size_mb = model_size_bytes / (1024 * 1024)  # Konversi ke MB

            # B. Hitung Jumlah Client yang berpartisipasi (N Clients)
            num_participating_clients = len(results)

            # C. Terapkan Rumus: Size x 2 (Up/Down) x Jumlah Client
            # Downlink: Server kirim model ke client di awal ronde
            # Uplink: Client kirim update ke server di akhir ronde
            round_cost_mb = model_size_mb * 2 * num_participating_clients

            # ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Akumulasi ke total
            self.total_communication_cost_mb += round_cost_mb

            # Simpan log untuk analisis nanti
            self.communication_log.append(
                {
                    "round": server_round,
                    "model_size_mb": model_size_mb,
                    "clients": num_participating_clients,
                    "round_cost_mb": round_cost_mb,
                    "total_cost_mb": self.total_communication_cost_mb,
                }
            )

            # Tampilkan di Terminal
            log(
                level=logging.INFO,
                msg=f"\n📊 [Round {server_round}] Communication Overhead Analysis:",
            )
            log(
                level=logging.INFO,
                msg=f"   - Model Size: {model_size_mb:.4f} MB",
            )
            log(
                level=logging.INFO,
                msg=f"   - Clients   : {num_participating_clients}",
            )
            log(
                level=logging.INFO,
                msg=f"   - Round Cost: {round_cost_mb:.4f} MB (Up+Down)",
            )
            log(
                level=logging.INFO,
                msg=f"   - 📈 TOTAL  : {self.total_communication_cost_mb:.4f} MB\n",
            )

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        return super().aggregate_evaluate(server_round, results, failures)
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if loss is not None and metrics:
            accuracy = metrics.get("accuracy", 0.0)

            log(
                level=logging.INFO,
                msg=f"🎯 [Round {server_round}] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%",
            )

        return loss, metrics

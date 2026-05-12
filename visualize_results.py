"""
Visualisasi Hasil Eksperimen Federated Learning
================================================
Script ini membaca file JSON hasil eksperimen dari folder experiment_results/
dan menghasilkan 5 grafik untuk laporan skripsi:

1. Akurasi per Ronde Komunikasi (Baseline vs Quantization)
2. Total Biaya Komunikasi untuk mencapai target akurasi
3. Hubungan RMSE Quantization Error dengan Akurasi Model Global
4. Variasi Jumlah Klien vs Akurasi Model Global
5. Variasi Jumlah Klien vs Total Biaya Komunikasi
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Utilitas: Load data eksperimen
# ============================================================

def load_experiment_data(experiment_dir="experiment_results"):
    """Load semua file JSON hasil eksperimen."""
    results = []
    pattern = os.path.join(experiment_dir, "experiment_*.json")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r") as f:
            data = json.load(f)
        results.append(data)
    return results


def find_experiment(results, scenario, num_clients=None):
    """Cari eksperimen berdasarkan skenario dan jumlah klien."""
    for r in results:
        cfg = r["experiment_config"]
        if cfg["scenario"] == scenario:
            if num_clients is None or cfg["num_clients"] == num_clients:
                return r
    return None


def find_all_experiments(results, scenario):
    """Cari semua eksperimen untuk skenario tertentu."""
    found = []
    for r in results:
        if r["experiment_config"]["scenario"] == scenario:
            found.append(r)
    return found


# ============================================================
# Grafik 1: Akurasi per Ronde Komunikasi
# ============================================================

def plot_accuracy_per_round(baseline_data, quant_data, save_path=None):
    """
    Grafik akurasi per ronde komunikasi.
    Membandingkan Baseline (FP32) vs Affine Quantization (INT8).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline
    rounds_b = [d["round"] for d in baseline_data["per_round_data"]]
    acc_b = [d["accuracy"] for d in baseline_data["per_round_data"]]
    ax.plot(rounds_b, acc_b, marker='o', markersize=3, linewidth=2,
            color='#1f77b4', label='Baseline (FP32)')

    # Quantization
    rounds_q = [d["round"] for d in quant_data["per_round_data"]]
    acc_q = [d["accuracy"] for d in quant_data["per_round_data"]]
    ax.plot(rounds_q, acc_q, marker='s', markersize=3, linewidth=2,
            color='#ff7f0e', label='Affine Quantization (INT8)')

    ax.set_xlabel('Communication Rounds', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison: Baseline vs Quantization',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, edgecolor='grey', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Grafik 1] Disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Grafik 2: Total Biaya Komunikasi
# ============================================================

def plot_total_communication_cost(baseline_data, quant_data, save_path=None):
    """
    Bar chart total biaya komunikasi untuk mencapai target akurasi.
    Target = min(convergence_acc_baseline, convergence_acc_quantization).
    """
    # Ambil convergence accuracy
    conv_acc_b = baseline_data["summary"]["convergence_accuracy"]
    conv_acc_q = quant_data["summary"]["convergence_accuracy"]
    target_accuracy = min(conv_acc_b, conv_acc_q)

    # Hitung biaya komunikasi untuk mencapai target
    def cost_to_target(data, target):
        for d in data["per_round_data"]:
            if d["accuracy"] >= target:
                return d["cumulative_comm_cost_mb"]
        # Jika tidak pernah mencapai target, return total
        return data["summary"]["total_comm_cost_mb"]

    cost_b = cost_to_target(baseline_data, target_accuracy)
    cost_q = cost_to_target(quant_data, target_accuracy)

    # Convert ke GB jika besar
    unit = "MB"
    cost_b_display = cost_b
    cost_q_display = cost_q
    if cost_b > 1024 or cost_q > 1024:
        cost_b_display = cost_b / 1024
        cost_q_display = cost_q / 1024
        unit = "GB"

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(1)
    width = 0.35

    rects1 = ax.bar(x - width/2, [cost_b_display], width,
                     label='Baseline (FP32)', color='#1f77b4', edgecolor='grey')
    rects2 = ax.bar(x + width/2, [cost_q_display], width,
                     label='Affine Quantization (INT8)', color='#ff7f0e', edgecolor='grey')

    ax.set_ylabel(f'Communication Cost ({unit})', fontsize=12)
    ax.set_title(f'Total Communication Cost Comparison\n'
                 f'(Target Accuracy: {target_accuracy*100:.1f}%)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Grafik 2] Disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Grafik 3: RMSE Quantization Error vs Akurasi
# ============================================================

def plot_rmse_vs_accuracy(quant_data, save_path=None):
    """
    Scatter plot + trend line: RMSE quantization error vs akurasi.
    Hanya untuk skenario Affine Quantization.
    """
    rmse_vals = []
    acc_vals = []
    for d in quant_data["per_round_data"]:
        if d["rmse_quantization_error"] is not None:
            rmse_vals.append(d["rmse_quantization_error"])
            acc_vals.append(d["accuracy"])

    if not rmse_vals:
        print("[Grafik 3] Tidak ada data RMSE untuk diplot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(rmse_vals, acc_vals, color='#1f77b4', s=40, zorder=5, label='Data points')

    # Trend line
    if len(rmse_vals) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(rmse_vals, acc_vals)
        x_line = np.linspace(min(rmse_vals), max(rmse_vals), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#1f77b4', linewidth=2,
                label=f'Trend line (R²={r_value**2:.3f})')

    ax.set_xlabel('RMSE Quantization Error', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('RMSE vs Accuracy (Federated Learning)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Grafik 3] Disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Grafik 4: Variasi Jumlah Klien vs Akurasi
# ============================================================

def plot_clients_vs_accuracy(results, client_variations, save_path=None):
    """
    Line/bar plot: jumlah klien vs rata-rata akurasi 5 ronde terakhir.
    """
    baseline_accs = []
    quant_accs = []

    for k in client_variations:
        b_data = find_experiment(results, "baseline", k)
        q_data = find_experiment(results, "quantization", k)

        if b_data:
            accs = [d["accuracy"] for d in b_data["per_round_data"]]
            last5 = accs[-5:] if len(accs) >= 5 else accs
            baseline_accs.append(sum(last5) / len(last5))
        else:
            baseline_accs.append(0)

        if q_data:
            accs = [d["accuracy"] for d in q_data["per_round_data"]]
            last5 = accs[-5:] if len(accs) >= 5 else accs
            quant_accs.append(sum(last5) / len(last5))
        else:
            quant_accs.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(client_variations))
    width = 0.35

    rects1 = ax.bar(x - width/2, baseline_accs, width,
                     label='Baseline (FP32)', color='#1f77b4',
                     edgecolor='grey', alpha=0.8)
    rects2 = ax.bar(x + width/2, quant_accs, width,
                     label='Affine Quantization (INT8)', color='#ff7f0e',
                     edgecolor='grey', alpha=0.8)

    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel('Average Accuracy (last 5 rounds)', fontsize=12)
    ax.set_title('Client Variation vs Global Model Accuracy',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in client_variations])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Grafik 4] Disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Grafik 5: Variasi Jumlah Klien vs Total Biaya Komunikasi
# ============================================================

def plot_clients_vs_comm_cost(results, client_variations, save_path=None):
    """
    Grouped bar chart: jumlah klien vs total biaya komunikasi
    untuk mencapai target akurasi konvergensi per K.
    """
    baseline_costs = []
    quant_costs = []

    for k in client_variations:
        b_data = find_experiment(results, "baseline", k)
        q_data = find_experiment(results, "quantization", k)

        if b_data and q_data:
            # Target = min(convergence_acc_baseline@K, convergence_acc_quantization@K)
            conv_b = b_data["summary"]["convergence_accuracy"]
            conv_q = q_data["summary"]["convergence_accuracy"]
            target = min(conv_b, conv_q)

            def cost_to_target(data, t):
                for d in data["per_round_data"]:
                    if d["accuracy"] >= t:
                        return d["cumulative_comm_cost_mb"]
                return data["summary"]["total_comm_cost_mb"]

            baseline_costs.append(cost_to_target(b_data, target))
            quant_costs.append(cost_to_target(q_data, target))
        else:
            baseline_costs.append(0)
            quant_costs.append(0)

    # Convert ke GB jika diperlukan
    unit = "MB"
    max_cost = max(max(baseline_costs, default=0), max(quant_costs, default=0))
    if max_cost > 1024:
        baseline_costs = [c / 1024 for c in baseline_costs]
        quant_costs = [c / 1024 for c in quant_costs]
        unit = "GB"

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(client_variations))
    width = 0.35

    rects1 = ax.bar(x - width/2, baseline_costs, width,
                     label='Baseline (FP32)', color='#1f77b4',
                     edgecolor='grey', alpha=0.8)
    rects2 = ax.bar(x + width/2, quant_costs, width,
                     label='Affine Quantization (INT8)', color='#ff7f0e',
                     edgecolor='grey', alpha=0.8)

    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel(f'Total Communication Cost ({unit})', fontsize=12)
    ax.set_title('Communication Cost Efficiency',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in client_variations])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Grafik 5] Disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Main: Generate semua grafik
# ============================================================

def main():
    """Generate semua grafik dari data eksperimen."""
    results = load_experiment_data()

    if not results:
        print("❌ Tidak ada data eksperimen ditemukan di folder experiment_results/")
        print("   Jalankan eksperimen terlebih dahulu menggunakan run_federated_learning.sh")
        return

    print(f"✓ Ditemukan {len(results)} file eksperimen")
    for r in results:
        cfg = r["experiment_config"]
        print(f"  - {cfg['scenario']}: K={cfg['num_clients']}, R={cfg['num_rounds']}")

    # Tentukan default num_clients untuk grafik 1-3
    default_k = 10
    baseline_main = find_experiment(results, "baseline", default_k)
    quant_main = find_experiment(results, "quantization", default_k)

    output_dir = "experiment_results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Grafik 1: Akurasi per Ronde
    if baseline_main and quant_main:
        print("\n📊 Generating Grafik 1: Akurasi per Ronde...")
        plot_accuracy_per_round(
            baseline_main, quant_main,
            save_path=os.path.join(output_dir, "grafik1_accuracy_per_round.png")
        )
    else:
        print(f"⚠ Data tidak lengkap untuk Grafik 1 (butuh baseline & quantization K={default_k})")

    # Grafik 2: Total Biaya Komunikasi
    if baseline_main and quant_main:
        print("\n📊 Generating Grafik 2: Total Biaya Komunikasi...")
        plot_total_communication_cost(
            baseline_main, quant_main,
            save_path=os.path.join(output_dir, "grafik2_total_comm_cost.png")
        )
    else:
        print(f"⚠ Data tidak lengkap untuk Grafik 2 (butuh baseline & quantization K={default_k})")

    # Grafik 3: RMSE vs Akurasi
    if quant_main:
        print("\n📊 Generating Grafik 3: RMSE vs Akurasi...")
        plot_rmse_vs_accuracy(
            quant_main,
            save_path=os.path.join(output_dir, "grafik3_rmse_vs_accuracy.png")
        )
    else:
        print(f"⚠ Data tidak ditemukan untuk Grafik 3 (butuh quantization K={default_k})")

    # Grafik 4 & 5: Variasi jumlah klien
    client_variations = [5, 10, 15]
    has_all = True
    for k in client_variations:
        if not find_experiment(results, "baseline", k):
            has_all = False
            print(f"⚠ Belum ada data baseline K={k}")
        if not find_experiment(results, "quantization", k):
            has_all = False
            print(f"⚠ Belum ada data quantization K={k}")

    if has_all:
        print("\n📊 Generating Grafik 4: Variasi Klien vs Akurasi...")
        plot_clients_vs_accuracy(
            results, client_variations,
            save_path=os.path.join(output_dir, "grafik4_clients_vs_accuracy.png")
        )

        print("\n📊 Generating Grafik 5: Variasi Klien vs Biaya Komunikasi...")
        plot_clients_vs_comm_cost(
            results, client_variations,
            save_path=os.path.join(output_dir, "grafik5_clients_vs_comm_cost.png")
        )
    else:
        print("\n⚠ Data variasi klien belum lengkap untuk Grafik 4 & 5")
        print(f"   Dibutuhkan: baseline & quantization untuk K={client_variations}")

    print("\n✅ Selesai!")


if __name__ == "__main__":
    main()

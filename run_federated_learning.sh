#!/bin/bash

# Warna untuk output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Buat direktori logs jika belum ada
LOG_DIR="./logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

clear

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Federated Learning Runner Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Menu untuk memilih mode kuantisasi
echo -e "${YELLOW}Pilih mode kuantisasi:${NC}"
echo "1) Tanpa Kuantisasi (quantization=none)"
echo "2) Dengan Kuantisasi (default)"
echo ""

read -p "Masukkan pilihan (1 atau 2): " quantization_choice

# Validasi pilihan kuantisasi
if [[ "$quantization_choice" != "1" && "$quantization_choice" != "2" ]]; then
    echo -e "${RED}❌ Pilihan tidak valid! Gunakan 1 atau 2${NC}"
    exit 1
fi

# Input jumlah server rounds
echo ""
read -p "Masukkan jumlah rounds (default: 1): " num_rounds
num_rounds=${num_rounds:-1}

# Input batch size
read -p "Masukkan batch size (default: 32): " batch_size
batch_size=${batch_size:-32}

# Input proporsi klien (fraction)
read -p "Masukkan proporsi klien (0.0 - 1.0, default: 1.0): " client_fraction
client_fraction=${client_fraction:-1.0}

# Input local epochs
read -p "Masukkan jumlah local epochs (default: 1): " local_epochs
local_epochs=${local_epochs:-1}

# Validasi input numerik
# num_rounds, batch_size, and local_epochs must be integers
for val in "$num_rounds" "$batch_size" "$local_epochs"; do
    if ! [[ "$val" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}❌ Round, Batch Size, dan Epoch harus berupa angka bulat! ($val)${NC}"
        exit 1
    fi
done

# client_fraction can be float
if ! [[ "$client_fraction" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "${RED}❌ Proporsi klien harus berupa angka atau desimal! ($client_fraction)${NC}"
    exit 1
fi

if [ "$num_rounds" -lt 1 ] || [ "$batch_size" -lt 1 ] || [ "$local_epochs" -lt 1 ]; then
    echo -e "${RED}❌ Nilai input minimal harus 1!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Konfigurasi:${NC}"
echo "  - Server Rounds: $num_rounds"
echo "  - Batch Size: $batch_size"
echo "  - Local Epochs: $local_epochs"
echo "  - Proporsi Klien: $client_fraction"

# Menentukan parameter konfigurasi
CONFIG_PARAMS="num-server-rounds=$num_rounds batch-size=$batch_size local-epochs=$local_epochs fraction-fit=$client_fraction fraction_evaluate=$client_fraction"

if [ "$quantization_choice" = "1" ]; then
    echo "  - Mode Kuantisasi: Tanpa Kuantisasi (quantization=none)"
    QUANT_LABEL="no-quant"
    CMD="flwr run -c '$CONFIG_PARAMS quantization=\"none\"' fl-thesis/. local-deployment"
else
    echo "  - Mode Kuantisasi: Dengan Kuantisasi"
    QUANT_LABEL="with-quant"
    CMD="flwr run -c '$CONFIG_PARAMS' fl-thesis/. local-deployment"
fi

# Buat nama file log dengan timestamp dan konfigurasi
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/fl_run_${QUANT_LABEL}_r${num_rounds}_${TIMESTAMP}.log"

echo ""
echo -e "${YELLOW}Perintah yang akan dijalankan:${NC}"
echo -e "${BLUE}$CMD${NC}"
echo ""
echo -e "${CYAN}📝 Output akan disimpan di: ${LOG_FILE}${NC}"
echo ""

# Konfirmasi sebelum menjalankan
read -p "Lanjutkan eksekusi? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo -e "${RED}❌ Dibatalkan${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}▶ Menjalankan Federated Learning...${NC}"
echo ""

# Jalankan perintah dan redirect output ke log file menggunakan tee
# 2>&1 untuk menangkap stderr juga
eval "$CMD" 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Federated Learning selesai dengan sukses!${NC}"
    echo -e "${CYAN}📁 Log file: ${LOG_FILE}${NC}"
else
    echo ""
    echo -e "${RED}❌ Terjadi kesalahan saat menjalankan Federated Learning${NC}"
    echo -e "${CYAN}📁 Log file: ${LOG_FILE}${NC}"
    exit 1
fi

#!/bin/bash

# Script untuk menjalankan Federated Learning dengan opsi kuantisasi
# Usage: ./run_federated_learning.sh
# Setiap sesi akan disimpan ke file log terpisah di folder 'logs/'

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
read -p "Masukkan jumlah server rounds (default: 1): " num_rounds
num_rounds=${num_rounds:-1}

# Validasi input rounds (harus berupa angka)
if ! [[ "$num_rounds" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}❌ Jumlah rounds harus berupa angka!${NC}"
    exit 1
fi

if [ "$num_rounds" -lt 1 ]; then
    echo -e "${RED}❌ Jumlah rounds harus minimal 1!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Konfigurasi:${NC}"
echo "  - Server Rounds: $num_rounds"

# Menentukan perintah berdasarkan pilihan kuantisasi dan membuat nama log
if [ "$quantization_choice" = "1" ]; then
    echo "  - Mode Kuantisasi: Tanpa Kuantisasi (quantization=none)"
    QUANT_LABEL="no-quant"
    CMD="flwr run -c 'num-server-rounds=$num_rounds quantization=\"none\"' fl-thesis/. local-deployment --stream"
else
    echo "  - Mode Kuantisasi: Dengan Kuantisasi"
    QUANT_LABEL="with-quant"
    CMD="flwr run -c 'num-server-rounds=$num_rounds' fl-thesis/. local-deployment --stream"
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

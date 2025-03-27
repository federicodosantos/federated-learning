import socket
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.signal import find_peaks
import pickle
import Crypto
from Crypto.Cipher import AES
import os
import wfdb

# Konfigurasi Klien
HOST = '127.0.0.1'  # Loopback address (harus sama dengan server)
PORT = 65432  # Port server (harus sama dengan server)
KEY = b'Sixteen byte key'  # Kunci Enkripsi (harus sama dengan server!)

# Definisi Model (Harus sama dengan server!)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 122, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# Fungsi untuk memuat dan memproses data MIT-BIH menggunakan wfdb
def load_mitbih_data(client_id, resample_size=128):
    """
    Memuat, memproses, dan mengekstrak fitur dari data MIT-BIH untuk klien tertentu.
    """
    # Ganti dengan path ke dataset MIT-BIH Anda
    record = wfdb.rdrecord('mitbih_database/100')  # Ganti dengan rekaman yang sesuai
    ecg_signal = record.p_signal[:, 0]  # Ambil hanya satu lead ECG

    # Simulasikan label acak karena dataset MIT-BIH tidak langsung menyertakan label
    labels = np.random.randint(0, 5, len(ecg_signal))

    # Pemisahan data antar klien
    if client_id == 1:
        ecg_signal = ecg_signal[:len(ecg_signal) // 2]
        labels = labels[:len(labels) // 2]
    else:
        ecg_signal = ecg_signal[len(ecg_signal) // 2:]
        labels = labels[len(labels) // 2:]

    # Normalisasi
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Deteksi Puncak R
    peaks, _ = find_peaks(ecg_signal, distance=50)

    # Ekstraksi Fitur
    features, feature_labels = [], []
    for peak_index in peaks:
        start = max(0, peak_index - resample_size // 2)
        end = min(len(ecg_signal), peak_index + resample_size // 2)
        window = ecg_signal[start:end]

        # Pastikan ukuran window tetap
        if len(window) < resample_size:
            padding = np.zeros(resample_size - len(window))
            window = np.concatenate([window, padding])
        elif len(window) > resample_size:
            window = window[:resample_size]

        features.append(window)
        feature_labels.append(labels[peak_index])

    return np.array(features), np.array(feature_labels)

# Fungsi untuk Mengenkripsi Data
def encrypt(data, key):
    cipher = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

# Fungsi untuk Mendekripsi Data
def decrypt(nonce, ciphertext, tag, key):
    cipher = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

# Fungsi untuk melatih model lokal
def train_local_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().unsqueeze(1)  # Sesuaikan dimensi untuk Conv1D
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

    return model

# Fungsi untuk menambahkan noise (Differential Privacy sederhana)
def add_noise(model, sensitivity, epsilon):
    """Menambahkan noise ke parameter model untuk differential privacy."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * (sensitivity / epsilon)
            param.add_(noise)
    return model

if __name__ == '__main__':
    client_id = 1  # Ubah sesuai klien
    print(f"Klien {client_id} dimulai...")

    # Muat Data Lokal
    features, labels = load_mitbih_data(client_id)

    # Ubah data menjadi format PyTorch
    X_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)

    # Buat DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Inisialisasi Model Lokal
    local_model = SimpleCNN()

    # Koneksi ke Server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Terhubung ke server di {HOST}:{PORT}")

            # Latih Model Lokal
            trained_model = train_local_model(local_model, train_loader)

            # Tambahkan Noise (Differential Privacy - Sederhana!)
            sensitivity = 0.1
            epsilon = 1.0
            trained_model = add_noise(trained_model, sensitivity, epsilon)

            # Kirim Model ke Server
            model_data = pickle.dumps(trained_model)
            nonce, ciphertext, tag = encrypt(model_data, KEY)

            # Kirim data yang sudah dienkripsi
            s.sendall(nonce)
            s.sendall(ciphertext)
            s.sendall(tag)

            # Terima model global dari server
            nonce = s.recv(16)
            ciphertext = s.recv(4096)
            tag = s.recv(16)
            decrypted_data = decrypt(nonce, ciphertext, tag, KEY)
            updated_global_model = pickle.loads(decrypted_data)
            print("Model global diterima dan diperbarui.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            s.close()
            print(f"Klien {client_id} selesai.")

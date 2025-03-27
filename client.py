import socket
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.signal import find_peaks
import pickle

# Konfigurasi Klien
HOST = '127.0.0.1'  # Loopback address
PORT = 65432  # Port server

# Definisi Model (Harus sama dengan server!)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_mitbih_data(client_id, resample_size=128):
    """
    Memuat, memproses, dan mengekstrak fitur dari data sintetis.
    """
    np.random.seed(client_id)
    
    # Buat data sintetis
    ecg_signal = np.random.randn(1000)
    labels = np.random.randint(0, 10, len(ecg_signal))

    # Normalisasi
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Deteksi Puncak 
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

def send_data(sock, data):
    """Kirim data dengan format panjang + data"""
    # Kirim panjang data (4 byte)
    data_length = len(data)
    sock.sendall(data_length.to_bytes(4, byteorder='big'))
    
    # Kirim data
    sock.sendall(data)

def receive_data(sock):
    """Terima data dengan format panjang + data"""
    # Terima panjang data
    length_bytes = sock.recv(4)
    data_length = int.from_bytes(length_bytes, byteorder='big')
    
    # Terima data
    return sock.recv(data_length)

def train_local_model(model, train_loader, epochs=10, learning_rate=0.001):
    """Latih model lokal"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.float().unsqueeze(1)
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

    return model

def main():
    client_id = 1
    print(f"Klien {client_id} dimulai...")

    # Muat Data Lokal
    features, labels = load_mitbih_data(client_id)

    # Persiapan data
    X_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    local_model = SimpleCNN()

    # Koneksi ke Server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Terhubung ke server di {HOST}:{PORT}")

            # Latih Model Lokal
            trained_model = train_local_model(local_model, train_loader)

            # Serialisasi model
            model_data = pickle.dumps(trained_model)
            
            # Kirim model
            send_data(s, model_data)

            # Terima model global
            received_data = receive_data(s)
            updated_global_model = pickle.loads(received_data)
            
            print("Model global diterima dan diperbarui.")

        except Exception as e:
            print(f"Error tidak terduga: {e}")

    print(f"Klien {client_id} selesai.")

if __name__ == '__main__':
    main()
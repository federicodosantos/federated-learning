import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import pickle
import os

# Konfigurasi Server
HOST = '0.0.0.0'  # Loopback address
PORT = 65432  # Port untuk mendengarkan

# Definisi Model Global
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

def save_model(model, filename='global_model.pkl'):
    """Simpan model ke file pickle"""
    try:
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        # Simpan model lengkap
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model berhasil disimpan di {filename}")
    except Exception as e:
        print(f"Kesalahan menyimpan model: {e}")

def send_large_data(sock, data):
    """Kirim data besar dengan pemeriksaan ukuran"""
    # Ubah ukuran data menjadi int32 (4 byte)
    data_size = len(data)
    sock.sendall(data_size.to_bytes(4, byteorder='big'))
    
    # Kirim data dalam bagian-bagian
    chunk_size = 4096
    for i in range(0, len(data), chunk_size):
        sock.sendall(data[i:i+chunk_size])

def recv_large_data(sock):
    """Terima data besar dengan pemeriksaan ukuran"""
    # Terima ukuran data
    data_size_bytes = sock.recv(4)
    data_size = int.from_bytes(data_size_bytes, byteorder='big')
    
    # Terima data dalam bagian-bagian
    data = bytearray()
    while len(data) < data_size:
        chunk = sock.recv(min(4096, data_size - len(data)))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data.extend(chunk)
    
    return bytes(data)

def federated_averaging(global_model, client_models, client_weights):
    """Lakukan federated averaging"""
    try:
        with torch.no_grad():
            global_params = OrderedDict(global_model.named_parameters())
            for name, param in global_params.items():
                param.data.zero_()
            
            for client_model, weight in zip(client_models, client_weights):
                client_params = OrderedDict(client_model.named_parameters())
                for name, param in global_params.items():
                    param.data += client_params[name].data * weight
        return global_model
    except Exception as e:
        print(f"Federated averaging error: {e}")
        return global_model

def handle_client(conn, addr, global_model, client_models, client_weights, client_index):
    """Tangani koneksi klien"""
    print(f"Terhubung oleh {addr}")
    try:
        # Terima model dari klien
        received_data = recv_large_data(conn)
        client_model = pickle.loads(received_data)
        client_models[client_index] = client_model
        
        # Lakukan Federated Averaging jika semua klien sudah terhubung
        if all(model is not None for model in client_models):
            updated_global_model = federated_averaging(global_model, client_models, client_weights)
            print("Model global telah diperbarui.")
            
            # Serialisasi model global
            model_data = pickle.dumps(updated_global_model)
            
            # Kirim model global
            send_large_data(conn, model_data)
            
            # Simpan model global ke file
            save_model(updated_global_model, f'global_model_round_{client_index + 1}.pkl')
    
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()

def main():
    # Inisialisasi Model Global
    global_model = SimpleCNN()
    
    # Inisialisasi Model Klien (sebagai placeholder)
    num_clients = 2
    client_models = [None] * num_clients
    client_weights = [0.5, 0.5]  # Bobot sama untuk setiap klien
    
    # Buat socket TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server mendengarkan di {HOST}:{PORT}")
        
        client_threads = []
        client_count = 0
        while client_count < num_clients:
            conn, addr = s.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr, global_model, client_models, client_weights, client_count)
            )
            client_thread.start()
            client_threads.append(client_thread)
            client_count += 1
        
        # Tunggu semua thread selesai
        for thread in client_threads:
            thread.join()
        
        print("Semua klien selesai, server ditutup.")
        
        # Simpan model global akhir
        save_model(global_model, 'final_global_model.pkl')

if __name__ == '__main__':
    main()
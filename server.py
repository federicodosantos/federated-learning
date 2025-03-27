import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import pickle
import Crypto
from Crypto.Cipher import AES
import os

# Konfigurasi Server
HOST = '0.0.0.0'  # Loopback address
PORT = 65432  # Port untuk mendengarkan

# Kunci Enkripsi (Harus sama dengan klien!)
KEY = b'Sixteen byte key'  # Ganti dengan kunci yang lebih kuat

# Definisi Model Global
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

# Fungsi untuk Mengenkripsi Data
def encrypt(data, key):
    cipher = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

# Fungsi untuk Mendekripsi Data
def decrypt(nonce, ciphertext, tag, key):
    cipher = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_EAX, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data

# Fungsi untuk menggabungkan pembaruan model (Federated Averaging)
def federated_averaging(global_model, client_models, client_weights):
    with torch.no_grad():
        global_params = OrderedDict(global_model.named_parameters())
        for name, param in global_params.items():
            param.data.zero_()
        
        for client_model, weight in zip(client_models, client_weights):
            client_params = OrderedDict(client_model.named_parameters())
            for name, param in global_params.items():
                param.data += client_params[name].data * weight
    return global_model

# Fungsi untuk menangani setiap koneksi klien
def handle_client(conn, addr, global_model, client_models, client_weights, client_index):
    print(f"Terhubung oleh {addr}")
    try:
        # Menerima model dari klien
        nonce = conn.recv(16)  # Panjang nonce AES
        ciphertext = conn.recv(4096)  # Ukuran buffer
        tag = conn.recv(16)  # Panjang tag MAC
        
        decrypted_data = decrypt(nonce, ciphertext, tag, KEY)
        client_model = pickle.loads(decrypted_data)
        client_models[client_index] = client_model
        
        # Lakukan Federated Averaging jika kita sudah menerima dari semua klien
        if all(model is not None for model in client_models):
            federated_averaging(global_model, client_models, client_weights)
            print("Model global telah diperbarui.")
            
            # Kirim kembali model global yang sudah diperbarui ke klien
            model_data = pickle.dumps(global_model)
            nonce, ciphertext, tag = encrypt(model_data, KEY)
            conn.sendall(nonce)
            conn.sendall(ciphertext)
            conn.sendall(tag)
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        conn.close()

def receive_data(conn, size):
    data = b''
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None  # Koneksi terputus
        data += packet
    return data        

if __name__ == '__main__':
    # Inisialisasi Model Global
    global_model = SimpleCNN()
    
    # Inisialisasi Model Klien (sebagai placeholder)
    num_clients = 2
    client_models = [None] * num_clients  # None berarti belum menerima model dari klien
    client_weights = [0.6, 0.4]
    
    # Buat socket TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server mendengarkan di {HOST}:{PORT}")
        
        client_count = 0
        while client_count < num_clients:
            conn, addr = s.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr, global_model, client_models, client_weights, client_count)
            )
            client_thread.start()
            client_count += 1
        
        print("Semua klien terhubung, menunggu pembaruan model...")

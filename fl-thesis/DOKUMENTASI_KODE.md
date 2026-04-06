# Dokumentasi Lengkap Codebase — Federated Learning dengan Asymmetric INT8 Quantization

> Dokumen ini menjelaskan **secara detail** setiap file dalam proyek `fl-thesis`, mulai dari arsitektur tingkat tinggi, alur eksekusi per ronde, hingga rumus matematika kuantisasi yang digunakan. Tujuannya agar Anda benar-benar memahami apa yang terjadi di setiap baris kode, bukan sekadar "yang penting jalan".

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Struktur Direktori](#2-struktur-direktori)
3. [Konfigurasi Proyek — `pyproject.toml`](#3-konfigurasi-proyek--pyprojecttoml)
4. [Modul Task — `task.py`](#4-modul-task--taskpy)
5. [Modul Kuantisasi — `quantization.py`](#5-modul-kuantisasi--quantizationpy)
6. [Aplikasi Klien — `client_app.py`](#6-aplikasi-klien--client_apppy)
7. [Strategi Server — `custom_strategy.py`](#7-strategi-server--custom_strategypy)
8. [Aplikasi Server — `server_app.py`](#8-aplikasi-server--server_apppy)
9. [Alur Eksekusi Lengkap (End-to-End per Ronde)](#9-alur-eksekusi-lengkap-end-to-end-per-ronde)
10. [Penjelasan Matematika Kuantisasi](#10-penjelasan-matematika-kuantisasi)
11. [Alur Serialisasi dan Deserialisasi Data](#11-alur-serialisasi-dan-deserialisasi-data)
12. [Pencatatan Biaya Komunikasi](#12-pencatatan-biaya-komunikasi)
13. [Masalah-Masalah yang Telah Diperbaiki](#13-masalah-masalah-yang-telah-diperbaiki)

---

## 1. Gambaran Umum Proyek

Proyek ini mengimplementasikan **Federated Learning (FL)** untuk klasifikasi kanker kulit (jinak vs. ganas) menggunakan dataset **PAD-UFES-20**. Model yang digunakan adalah **ResNet-18** (pretrained ImageNet). 

Fitur utama yang membedakan proyek ini dari FL standar adalah penerapan **Asymmetric Per-Channel INT8 Quantization** pada parameter model saat dikirim antara server dan klien. Tujuannya adalah mengurangi **biaya komunikasi** (ukuran data yang dikirim) hingga ~75% tanpa menurunkan akurasi model secara signifikan.

### Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Framework FL | [Flower (flwr)](https://flower.ai/) ≥ 1.21.0 |
| Deep Learning | PyTorch 2.7.1 |
| Model | ResNet-18 (torchvision) |
| Kuantisasi | [TorchAO](https://github.com/pytorch/ao) ≥ 0.15.0 |
| Dataset | PAD-UFES-20 (dermatologi) |
| Bahasa | Python 3.10 |

### Arsitektur Tingkat Tinggi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FLOWER SERVER                                  │
│                                                                         │
│  server_app.py ──► FedAvgWithCost (custom_strategy.py)                  │
│      │                    │                                              │
│      │          ┌─────────┴──────────┐                                   │
│      │          │  configure_fit()   │  Kuantisasi global model          │
│      │          │  aggregate_fit()   │  Dekuantisasi upload klien        │
│      │          │  aggregate_evaluate│  Agregasi FedAvg                  │
│      │          └────────────────────┘                                   │
│      │                    │                                              │
│      │            quantization.py                                        │
│      │          (quantize / dequantize)                                   │
└──────┼──────────────────────────────────────────────────────────────────┘
       │  gRPC / Flower Protocol
       │  ← Download: INT8 + metadata (scales, zero_points)
       │  → Upload:   INT8 + metadata (scales, zero_points)
┌──────┼──────────────────────────────────────────────────────────────────┐
│      │                       FLOWER CLIENT                              │
│      ▼                                                                  │
│  client_app.py ──► FlowerClient                                         │
│      │                    │                                              │
│      │          ┌─────────┴──────────┐                                   │
│      │          │  fit()             │  Training lokal                   │
│      │          │  evaluate()        │  Evaluasi lokal                   │
│      │          │  _maybe_dequant..  │  Dekuantisasi dari server         │
│      │          └────────────────────┘                                   │
│      │                    │                                              │
│      │            task.py                                                │
│      │    (load_model, load_data, train, test)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Struktur Direktori

```
fl-thesis/
├── pyproject.toml              # Konfigurasi proyek, dependensi, dan parameter FL
├── superexec.Dockerfile        # Dockerfile untuk deployment
│
├── fl_thesis/                  # Package utama
│   ├── __init__.py             # Marker package Python
│   ├── task.py                 # Model, dataset, training, dan evaluasi
│   ├── quantization.py         # Logika kuantisasi dan dekuantisasi (INT8)
│   ├── client_app.py           # Logika klien FL (FlowerClient)
│   ├── custom_strategy.py      # Strategi server FL (FedAvgWithCost)
│   ├── server_app.py           # Entry point server FL
│   ├── data_split.py           # Utility pembagian data antar klien
│   └── dataset/                # Folder dataset lokal
│
└── scripts/                    # Script utilitas
```

---

## 3. Konfigurasi Proyek — `pyproject.toml`

File ini adalah pusat konfigurasi seluruh proyek FL. Flower membaca file ini untuk mengetahui:
- Di mana `ServerApp` dan `ClientApp` berada
- Parameter eksperimen apa yang digunakan

### Bagian-bagian Penting

```toml
[tool.flwr.app.components]
serverapp = "fl_thesis.server_app:app"    # Entry point server
clientapp = "fl_thesis.client_app:app"    # Entry point klien
```

Flower akan memanggil objek `app` dari masing-masing modul saat dijalankan.

```toml
[tool.flwr.app.config]
num-server-rounds = 1           # Jumlah ronde FL
fraction-fit = 0.1              # Fraksi klien yang diundang untuk training
min_fit_clients = 1             # Minimum klien untuk training
fraction_evaluate = 0.1         # Fraksi klien untuk evaluasi
min_evaluate_clients = 1        # Minimum klien untuk evaluasi  
min_available_clients = 1       # Minimum klien online sebelum mulai
local-epochs = 1                # Jumlah epoch training lokal per ronde
quantization = "asymmetric"     # Jenis kuantisasi ("none" atau "asymmetric")
quantization-bits = 8           # Presisi kuantisasi (8-bit)
```

Semua nilai di atas dapat diakses oleh kode melalui `context.run_config["nama-key"]`.

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 1      # Jumlah klien (node) dalam simulasi
```

**Catatan penting:** `num-supernodes` menentukan berapa banyak klien virtual yang akan dibuat saat simulasi. Jika Anda ingin menguji dengan 3 klien, ubah nilai ini menjadi 3.

---

## 4. Modul Task — `task.py`

**Lokasi:** `fl_thesis/task.py`  
**Tanggung Jawab:** Menyediakan semua komponen ML dasar — model, dataset, fungsi training, dan evaluasi.

File ini adalah "fondasi" yang tidak tahu apa-apa tentang Federated Learning maupun kuantisasi. Ia murni berisi logika machine learning lokal.

### 4.1 Konstanta Global

```python
MANIFEST_FILE = "/app/data/metadata.csv"    # Path ke file CSV metadata gambar
IMAGE_FOLDER = "/app/data/images"            # Path ke folder gambar
DEVICE = torch.device("cpu")                 # Device komputasi
```

Path-path ini menunjuk ke lokasi data di dalam container Docker. Saat dijalankan di Docker, setiap klien akan memiliki volume data sendiri yang di-mount ke `/app/data/`.

### 4.2 `load_model()` → `ResNet`

```python
def load_model() -> ResNet:
```

**Apa yang dilakukan:**
1. Memuat model **ResNet-18** dengan bobot pretrained dari ImageNet (`ResNet18_Weights.DEFAULT`)
2. Mengganti *fully connected layer* terakhir (`model.fc`) dari 1000 kelas (ImageNet) menjadi **2 kelas** (jinak vs. ganas)

**Mengapa ResNet-18?**
- Cukup powerful untuk klasifikasi gambar medis
- Tidak terlalu besar sehingga cocok untuk FL (11.7 juta parameter)
- Transfer learning dari ImageNet mempercepat konvergensi

**Detail arsitektur ResNet-18:**
- Total parameter: ~11.17 juta
- Layer terakhir asli: `Linear(512, 1000)` → diganti menjadi `Linear(512, 2)`  
- Jumlah tensor (state_dict): **122 tensor** (termasuk weight, bias, running_mean, running_var, num_batches_tracked dari setiap BatchNorm layer)

### 4.3 `FederatedSkinLesionDataset` (class)

```python
class FederatedSkinLesionDataset(Dataset):
```

**Apa yang dilakukan:**
Kelas dataset kustom yang membaca:
- File CSV metadata (`metadata.csv`) yang berisi kolom `img_id` (nama file gambar) dan `binary_label` (0 = jinak, 1 = ganas)
- Folder gambar (`images/`) yang berisi file-file gambar dermatologi

**Alur `__getitem__(idx)`:**
1. Ambil nama file gambar dan label dari DataFrame pada baris ke-`idx`
2. Buka gambar menggunakan PIL (`Image.open`)
3. Konversi ke RGB (memastikan 3 channel)
4. Aplikasikan transformasi (resize, normalisasi)
5. Kembalikan tuple `(image_tensor, label)`

### 4.4 `load_data()` → `(trainloader, testloader)`

```python
def load_data():
```

**Apa yang dilakukan:**
1. Mendefinisikan pipeline transformasi gambar:
   - `Resize(224, 224)` — ResNet-18 membutuhkan input 224×224 piksel
   - `ToTensor()` — Konversi PIL Image ke PyTorch tensor (0-1)
   - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — Normalisasi menggunakan statistik ImageNet
2. Membuat instance `FederatedSkinLesionDataset`
3. Membagi dataset: **80% training, 20% validasi** menggunakan `random_split` dengan seed=42 (agar reproducible)
4. Membungkus menjadi `DataLoader` dengan `batch_size=32`

**Edge case yang ditangani:**
- Dataset kosong → kembalikan loader kosong
- Dataset < 5 sampel → gunakan semua untuk training, tidak ada validasi

### 4.5 `train(net, trainloader, epochs, device)` → `float`

```python
def train(net: ResNet, trainloader, epochs, device):
```

**Apa yang dilakukan:**
1. Pindahkan model ke device (CPU)
2. Setup `CrossEntropyLoss` sebagai loss function dan `Adam` optimizer dengan learning rate `1e-4`
3. Set model ke mode training (`net.train()`)
4. Loop setiap epoch:
   - Loop setiap batch:
     - Forward pass: `outputs = net(images)`
     - Hitung loss: `loss = criterion(outputs, labels)`
     - Backward pass: `loss.backward()`
     - Update weights: `optimizer.step()`
5. Kembalikan rata-rata loss training

**Return value:** `avg_trainloss` (float) — rata-rata loss selama training

### 4.6 `test(net, testloader, device)` → `(loss, accuracy)`

```python
def test(net: ResNet, testloader, device):
```

**Apa yang dilakukan:**
1. Set model ke mode evaluasi (`net.eval()`)
2. Tanpa gradient tracking (`torch.no_grad()`):
   - Loop setiap batch
   - Hitung loss dan jumlah prediksi benar
3. Kembalikan rata-rata loss dan akurasi

**Return value:** `(loss, accuracy)` — akurasi berupa proporsi (0.0 - 1.0)

### 4.7 Fungsi Utilitas Weights

```python
def get_weights(net: ResNet):
    """Mendapatkan weights model sebagai list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: ResNet, parameters):
    """Mengatur weights model dari list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
```

**`get_weights`**: Mengekstrak semua parameter model (termasuk BatchNorm statistics) sebagai list dari numpy array. Untuk ResNet-18 dengan 2 kelas output, akan menghasilkan **122 numpy arrays** dengan berbagai bentuk:
- Conv layer weight: `(64, 3, 7, 7)`, `(64, 64, 3, 3)`, `(512, 512, 3, 3)`, dll.  
- BatchNorm parameter: `(64,)`, `(128,)`, `(256,)`, `(512,)`
- Scalar: `()` (num_batches_tracked)
- FC layer: `(2, 512)` (weight), `(2,)` (bias)

**`set_weights`**: Kebalikannya — memasukkan kembali list numpy array ke dalam model PyTorch. `strict=True` memastikan **semua** key cocok (tidak ada yang hilang atau berlebih).

---

## 5. Modul Kuantisasi — `quantization.py`

**Lokasi:** `fl_thesis/quantization.py`  
**Tanggung Jawab:** Mengompresi parameter model float32 menjadi int8 (kuantisasi) dan mengembalikannya menjadi float32 (dekuantisasi).

Ini adalah **jantung** dari efisiensi komunikasi dalam proyek ini.

### 5.1 Konsep Dasar: Asymmetric Per-Channel Affine Quantization

**Apa itu kuantisasi?**
Kuantisasi adalah proses mengubah bilangan real (float32, 4 byte per angka) menjadi bilangan bulat kecil (int8, 1 byte per angka). Dengan begitu, data yang perlu dikirim berkurang ~75%.

**Mengapa "Asymmetric"?**
Karena rentang nilai tensor bisa tidak simetris terhadap nol. Misalnya, sebuah channel bisa memiliki nilai dari -0.5 hingga +3.0. Kuantisasi asimetris memungkinkan kita memetakan rentang miring ini secara optimal.

**Mengapa "Per-Channel"?**
Setiap channel (baris pertama tensor) mendapat parameter kuantisasi (`scale`, `zero_point`) sendiri. Ini lebih akurat daripada per-tensor karena distribusi nilai bisa sangat berbeda antar channel.

### 5.2 `_compute_affine_params_per_channel(tensor, bits)` → `(scales, zero_points)`

```python
def _compute_affine_params_per_channel(tensor: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray]:
```

**Apa yang dilakukan:**

Fungsi ini menghitung parameter kuantisasi (`scale` dan `zero_point`) untuk setiap channel dari sebuah tensor.

**Langkah-langkah:**

1. **Reshape ke 2D:**
   - Tensor skalar → `(1, 1)`
   - Tensor 1D, misalnya `(64,)` → `(1, 64)`
   - Tensor nD, misalnya `(64, 3, 7, 7)` → `(64, 147)` — baris pertama = channel

2. **Per setiap channel `i`:**
   - Cari `min_val` dan `max_val` dari channel tersebut
   - Jika `min_val == max_val` (channel konstan):
     - `scale = 1.0`, `zero_point = round(-min_val)`
   - Jika tidak:
     ```
     scale = (max_val - min_val) / (2^bits - 1)
     zero_point = round(-128 - min_val / scale)
     ```

**Mengapa rumusnya seperti itu?**

Rumus kuantisasi affine:
```
q = round(x / scale + zero_point)
x = (q - zero_point) * scale
```

Kita ingin memetakan `min_val` ke batas bawah INT8 (`-128`) dan `max_val` ke batas atas INT8 (`127`).

Dari persamaan:
```
-128 = min_val / scale + zero_point
 127 = max_val / scale + zero_point
```

Kita dapatkan:
```
scale = (max_val - min_val) / 255             ... (255 = 2^8 - 1)
zero_point = -128 - min_val / scale
```

**Return value:** `(scales, zero_points)` — masing-masing berupa numpy array 1D dengan panjang = jumlah channel.

### 5.3 `quantize_parameters(parameters, bits)` → `(quantized_bytes, params_list)`

```python
def quantize_parameters(
    parameters: List[np.ndarray],
    bits: int = 8,
) -> Tuple[List[bytes], List[Tuple[np.ndarray, np.ndarray]]]:
```

**Input:**
- `parameters`: List dari 122 numpy array float32 (semua parameter model ResNet-18)
- `bits`: Presisi kuantisasi (selalu 8 untuk INT8)

**Apa yang dilakukan per tensor:**

1. Pastikan dtype adalah float32
2. Simpan `original_shape` (misalnya `(64, 3, 7, 7)`)
3. Hitung `scales` dan `zero_points` per-channel menggunakan `_compute_affine_params_per_channel`
4. Reshape tensor ke 2D: `(n_channels, n_cols)`
5. **Per setiap channel `i`:**
   - Ambil data 1 baris: `ch_np = param_2d[i]`
   - Konversi ke PyTorch tensor: `ch_tensor = torch.from_numpy(ch_np.reshape(1, n_cols))`
   - Panggil **TorchAO** `quantize_affine()`:
     ```python
     q_ch = torchao_quantization.quantize_affine(
         input=ch_tensor,           # Data float32 1 channel
         block_size=(1, n_cols),     # Per-channel = seluruh baris
         scale=torch.tensor([scales[i]]),
         zero_point=torch.tensor([zero_points[i]]),
         output_dtype=torch.int8,   # Output: signed 8-bit integer
     )
     ```
   - Konversi hasil kembali ke numpy int8
   - Assertsion: ukuran output harus sama persis dengan `n_cols` (tidak boleh ada byte ekstra)
6. Reshape int8 2D kembali ke `original_shape`
7. Konversi ke raw bytes: `q_original.tobytes()`

**Output:**
- `quantized_bytes`: List berisi 122 objek `bytes`, masing-masing adalah representasi int8 dari tensor
- `params_list`: List berisi 122 tuple `(scales_array, zero_points_array)` — metadata yang dibutuhkan untuk dekuantisasi

**Contoh konkret:**
```
Tensor conv1.weight shape (64, 3, 7, 7):
  - Original: 64 × 3 × 7 × 7 × 4 bytes = 37,632 bytes (float32)
  - Quantized: 64 × 3 × 7 × 7 × 1 byte  = 9,408 bytes (int8)
  - Metadata: scales (64 float32) + zero_points (64 int32) = 512 bytes
  - Total: 9,920 bytes → penghematan ~74%
```

### 5.4 `dequantize_parameters(quantized_bytes, params_list, shapes, bits)` → `List[np.ndarray]`

```python
def dequantize_parameters(
    quantized_bytes: List[bytes],
    params_list: List[Tuple[np.ndarray, np.ndarray]],
    shapes: List[Tuple[int, ...]],
    bits: int = 8,
) -> List[np.ndarray]:
```

**Input:**
- `quantized_bytes`: List dari raw bytes int8
- `params_list`: List dari tuple `(scales, zero_points)` per tensor
- `shapes`: List dari shape asli setiap tensor (misalnya `[(64, 3, 7, 7), (64,), ...]`)
- `bits`: Presisi kuantisasi

**Apa yang dilakukan per tensor:**

1. Konversi bytes ke numpy int8: `q_array = np.frombuffer(q_bytes, dtype=np.int8)`
2. Reshape ke shape asli (validasi ketat — jika gagal, langsung error)
3. Reshape ke 2D: `(n_channels, n_cols)`
4. **Per setiap channel `i`:**
   - Ambil data int8: `q_ch_np = q_2d[i]`
   - Konversi ke PyTorch tensor
   - Panggil **TorchAO** `dequantize_affine()`:
     ```python
     ch_float = torchao_quantization.dequantize_affine(
         input=q_ch_tensor,          # Data int8
         block_size=(1, n_cols),
         scale=torch.tensor([scales[i]]),
         zero_point=torch.tensor([zero_points[i]]),
         input_dtype=torch.int8,
         output_dtype=torch.float32,  # Output: float32
     )
     ```
   - Jika TorchAO gagal, gunakan dekuantisasi manual:
     ```python
     result = (q_ch_np.astype(float32) - zero_point) * scale
     ```
5. Reshape kembali ke shape asli
6. Kembalikan sebagai float32 numpy array

**Output:** List dari 122 numpy array float32 — parameter model yang telah didekuantisasi.

---

## 6. Aplikasi Klien — `client_app.py`

**Lokasi:** `fl_thesis/client_app.py`  
**Tanggung Jawab:** Menjalankan logika klien FL — menerima model global, training lokal, dan mengirim kembali hasil.

### 6.1 Class `FlowerClient(NumPyClient)`

```python
class FlowerClient(NumPyClient):
```

`NumPyClient` adalah kelas dasar dari Flower yang menyederhanakan komunikasi menggunakan numpy arrays daripada objek protobuf mentah.

#### Inisialisasi (`__init__`)

```python
def __init__(self, net, trainloader, valloader, local_epochs,
             quantization="none", quantization_bits=8):
```

Menyimpan:
- `self.net` — model ResNet-18
- `self.trainloader`, `self.valloader` — data loader untuk training dan validasi
- `self.local_epochs` — jumlah epoch training lokal per ronde
- `self.quantization` — mode kuantisasi (`"none"` atau `"asymmetric"`)
- `self.quantization_bits` — bit presisi (8)
- `self.tensor_shapes` — **daftar shape semua parameter model** (122 shape). Ini penting karena setelah kuantisasi, kita perlu tahu shape asli untuk reshape kembali
- Pindahkan model ke device

#### 6.2 `fit(parameters, config)` — Training Lokal

```python
def fit(self, parameters, config):
```

Ini adalah fungsi inti yang dipanggil setiap ronde FL. Alurnya:

**Langkah 1: Terima & dekuantisasi parameter global**
```python
ndarrays = self._maybe_dequantize_from_server(parameters, config)
```
Parameter dari server mungkin dalam format int8 (terkuantisasi). Fungsi ini akan mendeteksi dan mendekuantisasi jika perlu.

**Langkah 2: Set weights & training lokal**
```python
set_weights(self.net, ndarrays)           # Masukkan parameter ke model
train_loss = train(self.net, ...)         # Training N epoch
weights = get_weights(self.net)           # Ambil parameter float32 hasil training
```

**Langkah 3: Kuantisasi sebelum upload (jika kuantisasi aktif)**
```python
if self.quantization != "none":
    quantized_bytes, params_list = quantize_parameters(weights, bits=self.quantization_bits)
```

Jika kuantisasi aktif:
1. Kuantisasi 122 tensor float32 → 122 objek bytes int8
2. Serialisasi metadata (`scales`, `zero_points`) ke JSON string
3. Log penghematan ukuran (misalnya "43,700 KB → 10,925 KB (75% reduction)")
4. Konversi bytes ke numpy int8 arrays dengan reshape ke shape asli
5. Kirim ke server sebagai tuple:
   ```python
   return (
       quantized_ndarrays,          # List[np.ndarray] dtype=int8
       len(dataset),                 # Jumlah sampel training
       {
           "train_loss": ...,
           "quantization_scales": "[[0.023, 0.015, ...], ...]",     # JSON
           "quantization_zero_points": "[[-128, -127, ...], ...]",  # JSON
       },
   )
   ```

Jika kuantisasi tidak aktif:
```python
return (weights, len(dataset), {"train_loss": ...})  # Float32 biasa
```

#### 6.3 `evaluate(parameters, config)` — Evaluasi Lokal

```python
def evaluate(self, parameters, config):
```

Alur sederhana:
1. Dekuantisasi parameter dari server
2. Set weights ke model
3. Jalankan evaluasi menggunakan `test()`
4. Kembalikan `(loss, jumlah_sampel, {"accuracy": ...})`

#### 6.4 `_maybe_dequantize_from_server(parameters, config)` — Dekuantisasi Download

```python
def _maybe_dequantize_from_server(self, parameters, config):
```

Fungsi ini menangani **dua jenis input** dari server:

**Kasus 1: Objek `Parameters` dari Flower (memiliki atribut `.tensors`)**
```python
if hasattr(parameters, "tensors"):
    ndarrays = parameters_to_ndarrays(parameters)
```
`parameters_to_ndarrays()` dari Flower secara otomatis melakukan:
- Iterasi setiap `bytes` di dalam `parameters.tensors`
- Membaca format `.npy` (membuang 128-byte header NumPy)
- Mengembalikan numpy array yang bersih

**Kasus 2: List Python biasa (fallback)**
```python
elif isinstance(parameters, (list, tuple)):
    for t in parameters:
        if isinstance(t, np.ndarray): ...
        elif isinstance(t, (bytes, bytearray)): ...
```

**Setelah mendapatkan `ndarrays`:**
Jika config mengandung metadata kuantisasi (`quantization_scales`), maka klien tahu bahwa data ini masih dalam format int8 dan perlu didekuantisasi:

```python
if self.quantization != "none" and "quantization_scales" in config:
    # Parse metadata JSON
    scales_list = json.loads(config["quantization_scales"])
    zero_points_list = json.loads(config["quantization_zero_points"])
    
    # Konversi ke format yang diterima dequantize_parameters
    q_bytes_list = [arr.tobytes() for arr in ndarrays]
    shapes = [arr.shape for arr in ndarrays]
    
    # Dekuantisasi: int8 → float32
    ndarrays = dequantize_parameters(q_bytes_list, params_list, shapes, bits=8)
```

#### 6.5 `client_fn(context)` — Factory Function

```python
def client_fn(context: Context):
```

Flower memanggil fungsi ini untuk **membuat instance klien baru**. Setiap kali Flower membutuhkan klien, ia memanggil `client_fn` yang:
1. Memuat model (`load_model()`)
2. Memuat data (`load_data()`)
3. Membaca konfigurasi dari `context.run_config`
4. Membuat dan mengembalikan `FlowerClient`

```python
app = ClientApp(client_fn)  # Registrasi ke Flower
```

---

## 7. Strategi Server — `custom_strategy.py`

**Lokasi:** `fl_thesis/custom_strategy.py`  
**Tanggung Jawab:** Mengelola alur FL di sisi server — kuantisasi model global, dekuantisasi upload klien, agregasi FedAvg, dan pencatatan biaya komunikasi.

### 7.1 Class `FedAvgWithCost(FedAvg)`

Mewarisi dari `FedAvg` (Federated Averaging) bawaan Flower dan menambahkan:
- Kuantisasi/dekuantisasi otomatis
- Pelacakan biaya komunikasi (upload + download dalam MB)

#### Inisialisasi

```python
def __init__(self, *args, quantization="none", quantization_bits=8, shapes=None, **kwargs):
```

Atribut penting:
- `self.quantization` — `"none"` atau `"asymmetric"`
- `self.quantization_bits` — 8
- `self.shapes` — cache shape tensor (diperlukan untuk dekuantisasi upload)
- `self.last_aggregated_ndarrays` — parameter float32 terakhir setelah agregasi (untuk perbandingan savings)
- `self.total_upload_cost_mb` / `self.total_download_cost_mb` — akumulasi biaya komunikasi
- `self.communication_log` — log per ronde

### 7.2 `configure_fit(server_round, parameters, client_manager)` — Broadcast ke Klien

```python
def configure_fit(self, server_round, parameters, client_manager):
```

Ini dipanggil **sebelum** klien mulai training. Tujuannya: menyiapkan model global + config untuk dikirim ke klien.

**Alur jika kuantisasi aktif:**

```
Parameter float32 (dari ronde sebelumnya atau inisialisasi)
    │
    ▼
parameters_to_ndarrays() ──► List[np.ndarray] float32 (122 arrays)
    │
    ▼
quantize_parameters() ──► quantized_bytes (List[bytes]) + params_list (metadata)
    │
    ▼
Serialisasi metadata ke JSON:
  - scales_serialized = "[[0.023, 0.015, ...], ...]"
  - zero_points_serialized = "[[-128, -127, ...], ...]"
  - shapes_serialized = "[[64, 3, 7, 7], [64], ...]"
    │
    ▼
Konversi bytes → int8 ndarrays → ndarrays_to_parameters() ──► Parameters (Flower object)
    │
    ▼
Inject metadata ke FitIns.config untuk setiap klien
```

**Detail kritis:**
```python
# Sinkronisasi shapes — server menyimpan shape agar bisa digunakan saat dekuantisasi upload klien
self.shapes = [arr.shape for arr in current_ndarrays]
```

Server menyimpan daftar shape di `self.shapes`. Ini PENTING karena saat klien mengirim kembali parameter terkuantisasi, server perlu tahu shape asli untuk reshape bytes → tensor.

**Injeksi config:**
```python
config.update({
    "quantization_scales": scales_serialized,       # Metadata per-channel
    "quantization_zero_points": zero_points_serialized,
    "quantization_shapes": shapes_serialized,       # Shape asli tiap tensor
    "quantization": "asymmetric",                   # Jenis kuantisasi
    "quantization_bits": self.quantization_bits,     # 8
})
```

Semua metadata ini dikirim bersama parameter model ke klien melalui `FitIns.config`, yang merupakan dictionary key-value.

### 7.3 `aggregate_fit(server_round, results, failures)` — Agregasi Upload

```python
def aggregate_fit(self, server_round, results, failures):
```

Dipanggil **setelah** semua klien selesai training dan mengirim kembali hasilnya.

**Alur:**

```
results = [(client1, FitRes1), (client2, FitRes2), ...]
    │
    ▼  (1) Log biaya upload SEBELUM dekuantisasi
_log_upload_cost()  ──► "Upload: 10.6838 MB/client × N clients (75% hemat)"
    │                    ↑ Ukuran yang dilog adalah payload int8 ASLI yang dikirim klien
    │
    ▼  (2) Dekuantisasi upload klien
_dequantize_fit_results()  ──► results dengan parameter float32
    │
    ▼  (3) Agregasi FedAvg (rata-rata tertimbang berdasarkan jumlah sampel)
super().aggregate_fit()  ──► aggregated_parameters
    │
    ▼  (4) Simpan parameter teragregasi untuk perbandingan savings di ronde berikutnya
self.last_aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
```

**Mengapa `_log_upload_cost()` dipanggil SEBELUM `_dequantize_fit_results()`?**
Karena setelah dekuantisasi, parameter dalam `results` berubah dari int8 (10.6 MB) menjadi float32 (42.7 MB). Jika kita log setelah dekuantisasi, angkanya akan salah (4× lebih besar dari kenyataan).

### 7.4 `_dequantize_fit_results(results)` — Dekuantisasi Upload

```python
def _dequantize_fit_results(self, results):
```

**Per setiap hasil klien:**

1. Cek apakah klien mengirim metadata kuantisasi (`quantization_scales` di metrics)
2. Jika tidak ada → asumsikan float32, lewati
3. Jika ada:
   - Parse metadata JSON dari `fit_res.metrics`
   - **Gunakan `parameters_to_ndarrays(fit_res.parameters)`** — ini PENTING karena:
     - Flower menyimpan numpy arrays menggunakan format `.npy` (dengan 128-byte header)
     - `parameters_to_ndarrays` secara otomatis membaca format ini dan menstrip header
     - Jika kita langsung mengakses `fit_res.parameters.tensors` (raw bytes), kita akan mendapat data yang 128 byte lebih banyak per tensor
   - Konversi numpy arrays ke raw bytes
   - Panggil `dequantize_parameters()` untuk mengubah int8 → float32
   - Buat `FitRes` baru dengan parameter float32 dan metrics bersih (tanpa metadata kuantisasi)

### 7.5 Fungsi Pencatatan Biaya Komunikasi

#### `_log_upload_cost(server_round, results)`
Menghitung total bytes dari semua tensor yang dikirim klien ke server, kemudian menampilkan:
- Ukuran per klien dalam MB
- Persentase penghematan dibanding float32

#### `_log_download_cost(server_round, tensor_bytes)`
Menghitung total bytes dari parameter yang di-broadcast server ke klien, kemudian menampilkan:
- Ukuran download per klien dalam MB
- Biaya kumulatif upload + download

### 7.6 `aggregate_evaluate(server_round, results, failures)` — Agregasi Evaluasi

Memanggil `super().aggregate_evaluate()` (FedAvg bawaan) dan menambahkan log akurasi rata-rata:
```
[Round 10] Loss: 0.6981 | Accuracy: 0.45%
```

---

## 8. Aplikasi Server — `server_app.py`

**Lokasi:** `fl_thesis/server_app.py`  
**Tanggung Jawab:** Titik masuk (entry point) untuk server FL. Menginisialisasi model dan strategi.

### 8.1 `weighted_average_metrics(metrics)` → `Metrics`

```python
def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
```

Fungsi agregasi metrik evaluasi. Menghitung **rata-rata akurasi tertimbang** berdasarkan jumlah sampel dari setiap klien:

```
weighted_accuracy = Σ(num_examples_i × accuracy_i) / Σ(num_examples_i)
```

Contoh: Jika klien A punya 100 sampel dengan akurasi 80% dan klien B punya 50 sampel dengan akurasi 60%:
```
weighted = (100 × 0.8 + 50 × 0.6) / (100 + 50) = 110 / 150 = 0.733
```

### 8.2 `server_fn(context)` — Factory Function Server

```python
def server_fn(context: Context):
```

1. Baca semua konfigurasi dari `context.run_config` (yang berasal dari `pyproject.toml`)
2. **Inisialisasi model:**
   ```python
   model = load_model()                          # ResNet-18 pretrained
   ndarrays = get_weights(model)                  # List[np.ndarray] float32
   parameters = ndarrays_to_parameters(ndarrays)  # Konversi ke format Flower
   ```
   Model diinisialisasi di server agar semua klien mulai dari titik yang sama (pretrained ImageNet weights).
3. **Buat strategi:**
   ```python
   strategy = FedAvgWithCost(
       fraction_fit=...,            # Fraksi klien untuk training
       initial_parameters=parameters, # Parameter awal (pretrained)
       evaluate_metrics_aggregation_fn=weighted_average_metrics,
       quantization=quantization,   # "asymmetric" atau "none"
       quantization_bits=8,
   )
   ```
4. **Konfigurasi server:**
   ```python
   config = ServerConfig(num_rounds=num_rounds)
   return ServerAppComponents(strategy=strategy, config=config)
   ```

```python
app = ServerApp(server_fn=server_fn)  # Registrasi ke Flower
```

---

## 9. Alur Eksekusi Lengkap (End-to-End per Ronde)

Berikut adalah alur **lengkap** satu ronde FL dengan kuantisasi aktif:

```
╔══════════════════════════════════════════════════════════════════╗
║                    RONDE n DIMULAI                               ║
╚══════════════════════════════════════════════════════════════════╝

[SERVER] configure_fit() dipanggil
    │
    ├─ 1. Ambil parameter global float32 (dari ronde sebelumnya atau inisialisasi)
    │     parameters_to_ndarrays(parameters) → 122 numpy arrays float32
    │
    ├─ 2. KUANTISASI (float32 → int8)
    │     quantize_parameters(ndarrays, bits=8) 
    │       → quantized_bytes: 122 bytes objects (total ~10.7 MB)
    │       → params_list: 122 tuples (scales, zero_points)
    │
    ├─ 3. Simpan shapes: self.shapes = [(64,3,7,7), (64,), ...]
    │
    ├─ 4. Konversi ke Flower Parameters
    │     np.frombuffer(q, dtype=int8).reshape(shape) → int8 ndarrays
    │     ndarrays_to_parameters(quantized_ndarrays) → Parameters object
    │       └─ Internal: setiap array di-serialize via np.save (menambah 128-byte header)
    │
    ├─ 5. Siapkan config metadata (JSON):
    │     {"quantization_scales": "[[...],...]", 
    │      "quantization_zero_points": "[[...],...]",
    │      "quantization_shapes": "[[64,3,7,7],...]"}
    │
    ├─ 6. Log download cost: ~10.67 MB (75% hemat)
    │
    └─ 7. Kirim FitIns(parameters=int8_params, config=metadata) ke setiap klien

═══════════════════ JARINGAN (Download ~10.7 MB) ═══════════════════

[CLIENT] fit(parameters, config) dipanggil
    │
    ├─ 1. DEKUANTISASI DOWNLOAD (int8 → float32)
    │     _maybe_dequantize_from_server(parameters, config)
    │       ├─ parameters_to_ndarrays(parameters) → int8 ndarrays
    │       │    └─ Flower membaca format .npy, membuang 128-byte header
    │       ├─ Parse metadata dari config (scales, zero_points)
    │       └─ dequantize_parameters(bytes, params_list, shapes) → float32 ndarrays
    │
    ├─ 2. SET WEIGHTS ke model
    │     set_weights(self.net, ndarrays)
    │
    ├─ 3. TRAINING LOKAL
    │     train(self.net, self.trainloader, epochs, device)
    │       └─ Adam optimizer, lr=1e-4, CrossEntropyLoss
    │
    ├─ 4. AMBIL WEIGHTS HASIL TRAINING
    │     weights = get_weights(self.net) → float32 ndarrays
    │
    ├─ 5. KUANTISASI UPLOAD (float32 → int8)
    │     quantize_parameters(weights, bits=8)
    │       → quantized_bytes + params_list (metadata)
    │
    ├─ 6. Serialisasi metadata ke JSON
    │
    └─ 7. Return ke server:
          (int8_ndarrays, num_examples, 
           {"train_loss": ..., "quantization_scales": "...", "quantization_zero_points": "..."})

═══════════════════ JARINGAN (Upload ~10.7 MB) ═══════════════════

[SERVER] aggregate_fit() dipanggil
    │
    ├─ 1. LOG UPLOAD COST (dari payload int8 ASLI, sebelum dekuantisasi)
    │     _log_upload_cost() → "Upload: 10.68 MB/client (75% hemat)"
    │
    ├─ 2. DEKUANTISASI UPLOAD (int8 → float32)
    │     _dequantize_fit_results(results)
    │       ├─ Parse metadata JSON dari metrics
    │       ├─ parameters_to_ndarrays(fit_res.parameters) → int8 ndarrays
    │       │    └─ Membersihkan 128-byte .npy header secara otomatis
    │       └─ dequantize_parameters() → float32 ndarrays
    │
    ├─ 3. AGREGASI FEDAVG
    │     super().aggregate_fit()
    │       └─ Rata-rata tertimbang parameter dari semua klien
    │          new_param[i] = Σ(num_examples_k × param_k[i]) / Σ(num_examples_k)
    │
    └─ 4. Simpan hasil agregasi untuk ronde berikutnya
          self.last_aggregated_ndarrays = float32 ndarrays

[SERVER] configure_evaluate() dipanggil → kirim model teragregasi ke klien untuk evaluasi

[CLIENT] evaluate(parameters, config) dipanggil
    │
    ├─ Dekuantisasi parameter
    ├─ Set weights
    ├─ test(net, valloader) → (loss, accuracy)
    └─ Return (loss, num_examples, {"accuracy": ...})

[SERVER] aggregate_evaluate() dipanggil
    │
    └─ weighted_average_metrics() → akurasi rata-rata tertimbang
       Log: "[Round n] Loss: ... | Accuracy: ...%"

╔══════════════════════════════════════════════════════════════════╗
║                    RONDE n SELESAI                               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 10. Penjelasan Matematika Kuantisasi

### 10.1 Affine Quantization (Kuantisasi Affine)

**Rumus kuantisasi (float → int):**
```
q = clamp(round(x / scale + zero_point), qmin, qmax)
```

**Rumus dekuantisasi (int → float):**
```
x̂ = (q - zero_point) × scale
```

Di mana:
- `x` = nilai float32 asli
- `q` = nilai int8 terkuantisasi
- `scale` = faktor skala (float32)
- `zero_point` = titik nol (int32)
- `qmin = -128`, `qmax = 127` (untuk signed int8)
- `x̂` = nilai float32 hasil rekonstruksi (mendekati `x`, tapi tidak persis sama)

### 10.2 Perhitungan scale dan zero_point

Diberikan sebuah channel dengan `min_val` dan `max_val`:

```
scale = (max_val - min_val) / (2^8 - 1) = (max_val - min_val) / 255
```

```
zero_point = round(-128 - min_val / scale)
```

**Intuisi:** Kita ingin memetakan:
- `min_val` → `-128` (batas bawah int8)
- `max_val` → `+127` (batas atas int8)

Sehingga seluruh rentang 256 level kuantisasi dimanfaatkan secara optimal.

### 10.3 Contoh Numerik

Misalkan sebuah channel memiliki nilai dari -0.5 hingga +1.5:

```
scale = (1.5 - (-0.5)) / 255 = 2.0 / 255 ≈ 0.00784

zero_point = round(-128 - (-0.5) / 0.00784) = round(-128 + 63.78) = round(-64.22) = -64
```

Kuantisasi nilai `x = 0.7`:
```
q = round(0.7 / 0.00784 + (-64)) = round(89.29 - 64) = round(25.29) = 25
```

Dekuantisasi `q = 25` kembali:
```
x̂ = (25 - (-64)) × 0.00784 = 89 × 0.00784 = 0.6978
```

Error: `|0.7 - 0.6978| = 0.0022` → sangat kecil!

### 10.4 Mengapa Per-Channel Lebih Baik?

Setiap channel (neuron output) dalam sebuah convolutional layer bisa memiliki distribusi nilai yang sangat berbeda:

```
Channel 0: [-0.01, +0.01]   → scale kecil, presisi tinggi
Channel 1: [-2.5, +3.0]     → scale besar, presisi lebih rendah  
```

Jika kita menggunakan **satu** scale untuk seluruh tensor (per-tensor), maka channel 0 akan "terbuang" karena hanya menggunakan sebagian kecil dari 256 level. Dengan per-channel, setiap channel mendapat scale optimalnya sendiri.

---

## 11. Alur Serialisasi dan Deserialisasi Data

### 11.1 Flower's Internal Serialization

Ketika Flower mengkonversi numpy array ke format Parameters (via `ndarrays_to_parameters`), secara internal ia memanggil `np.save()` untuk setiap array. Format `.npy` yang dihasilkan memiliki struktur:

```
[128-byte header][raw data bytes]
```

Header 128 byte ini berisi metadata NumPy (magic string, versi, shape, dtype, dll).

Saat menerima kembali, `parameters_to_ndarrays()` memanggil `np.load()` yang secara otomatis membaca dan membuang header ini, mengembalikan array yang bersih.

**Kesalahan fatal yang sudah diperbaiki:** Jika kita langsung mengakses `fit_res.parameters.tensors` (raw bytes), kita mendapat bytes **dengan** header. Menggunakan `np.frombuffer(raw_bytes, dtype=np.int8)` pada data ini menghasilkan array dengan 128 elemen ekstra (karena header 128 byte ÷ 1 byte per int8 = 128 elemen tambahan).

### 11.2 Alur Serialisasi Metadata Kuantisasi

Metadata kuantisasi (scales dan zero_points) dikirim melalui **config dictionary** (server → klien) atau **metrics dictionary** (klien → server), bukan melalui parameter tensor.

```
Server → Client:
  FitIns.config = {
      "quantization_scales": "[[0.023, 0.015], [0.008], ...]",        # JSON
      "quantization_zero_points": "[[-64, -58], [-128], ...]",        # JSON
      "quantization_shapes": "[[64, 3, 7, 7], [64], ...]",           # JSON
      "quantization": "asymmetric",
      "quantization_bits": 8
  }

Client → Server:
  FitRes.metrics = {
      "train_loss": 0.45,
      "quantization_scales": "[[0.021, 0.014], [0.009], ...]",        # JSON
      "quantization_zero_points": "[[-65, -57], [-127], ...]"         # JSON
  }
```

---

## 12. Pencatatan Biaya Komunikasi

### 12.1 Download Cost (Server → Client)

Dihitung di `configure_fit()` berdasarkan `quantized_bytes` (raw payload tanpa header Flower):

```python
download_mb = sum(len(t) for t in tensor_bytes) / (1024 * 1024)
```

### 12.2 Upload Cost (Client → Server)

Dihitung di `aggregate_fit()` **sebelum** dekuantisasi, berdasarkan `fit_res.parameters.tensors`:

```python
upload_bytes_total = sum(len(t) for t in fit_res.parameters.tensors)
```

**Catatan:** Nilai ini termasuk 128-byte header per tensor (total ~15.5 KB overhead untuk 122 tensor), tapi ini diabaikan karena relatif kecil dibanding total payload (~10.7 MB).

### 12.3 Persentase Penghematan

```python
savings = (1.0 - quantized_mb / float32_mb) * 100
```

Secara teori, kuantisasi float32 → int8 menghasilkan penghematan ~75% (4 byte → 1 byte per elemen).

---

## 13. Masalah-Masalah yang Telah Diperbaiki

### 13.1 Bug +128 Elemen Ekstra

**Akar Masalah:** Flower's `ndarrays_to_parameters` menggunakan `np.save()` yang menambahkan 128-byte header `.npy`. Kode lama mengakses `fit_res.parameters.tensors` secara langsung (raw bytes dengan header) lalu memanggil `np.frombuffer(dtype=np.int8)`, menghasilkan 128 elemen int8 ekstra.

**Solusi:** Gunakan `parameters_to_ndarrays(fit_res.parameters)` yang secara otomatis membersihkan header.

### 13.2 Bug Overwrite Variabel di Debug Logging

**Akar Masalah:** Blok debug logging kedua di `_dequantize_fit_results()` secara tidak sengaja menulis ulang variabel `quantized_bytes` dengan `fit_res.parameters.tensors` (data mentah + header), menimpa data bersih yang sudah diekstrak sebelumnya.

**Solusi:** Hapus baris overwrite `quantized_bytes = fit_res.parameters.tensors`.

### 13.3 Bug Upload Cost Salah Hitung

**Akar Masalah:** `_log_upload_cost()` dipanggil **setelah** `_dequantize_fit_results()`, sehingga ia menghitung ukuran parameter float32 hasil dekuantisasi (42.7 MB) bukan payload int8 asli (10.7 MB).

**Solusi:** Pindahkan `_log_upload_cost()` agar dipanggil **sebelum** dekuantisasi.

### 13.4 Bug Zero Point Overflow (INT8 vs UINT8)

**Akar Masalah:** Rumus zero_point lama: `zero_point = round(-min_val / scale)` memetakan `min_val` ke 0 (asumsi unsigned uint8, range 0-255). Namun, output TorchAO menggunakan signed int8 (range -128 s.d 127). Akibatnya, semua bobot positif yang dipetakan ke nilai > 127 mengalami **integer overflow**, berubah menjadi bilangan negatif besar, menghancurkan parameter model.

**Solusi:** Perbaiki rumus menjadi `zero_point = round(-128 - min_val / scale)` untuk memetakan ke rentang signed int8 yang benar.

**Dampak:** Akurasi model naik dari ~45% (stuck di sekitar level random) ke level yang sebanding dengan baseline tanpa kuantisasi.

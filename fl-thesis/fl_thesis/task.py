"""fl-thesis: A Flower / PyTorch app (Diadaptasi untuk PAD-UFES-20)."""

import logging
import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from flwr.common import log
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import ResNet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# =============================================================================
# Pengaturan Global (Sesuaikan path ini)
# =============================================================================
MANIFEST_FILE = "/app/data/metadata.csv"
IMAGE_FOLDER = "/app/data/images"
DEVICE = torch.device("cpu")


# =============================================================================
# 1. Model (Diganti dari Net ke ResNet-18)
# =============================================================================
def load_model() -> ResNet:
    """Memuat ResNet-18 dan mengganti layer terakhir untuk klasifikasi biner."""
    log(level=logging.INFO, msg="Memuat model ResNet-18 yang telah dipra-latih...")

    # Gunakan pretrained=True untuk transfer learning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Ganti 'fully connected' layer terakhir
    num_ftrs = model.fc.in_features
    # Kita butuh 2 output (Jinak vs. Ganas)
    model.fc = nn.Linear(num_ftrs, 2)

    log(level=logging.INFO, msg="Model siap digunakan.")
    return model


class FederatedSkinLesionDataset(Dataset):
    """Dataset kustom untuk PAD-UFES-20 di Named Volume."""

    def __init__(self, manifest_path, images_dir, transform=None):
        self.df = pd.read_csv(manifest_path)

        # Karena setiap klien sudah punya dataset masing-masing, tidak perlu filter
        self.client_data = self.df.reset_index(drop=True)

        print(f"Jumlah sample: {len(self.client_data)}")
        print(self.client_data.head())

        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.client_data)

    def __getitem__(self, idx):
        img_name = self.client_data.iloc[idx]["img_id"]
        label = self.client_data.iloc[idx]["binary_label"]

        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading {img_name}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label


# Fungsi load_data yang dimodifikasi total
def load_data(batch_size: int = 32):
    """Memuat partisi data PAD-UFES-20."""
    log(level=logging.INFO, msg="Memuat data untuk klien saat ini...")

    # Definisikan transformasi untuk ResNet-18
    pytorch_transforms = Compose(
        [
            Resize((224, 224)),  # ResNet butuh 224x224
            ToTensor(),
            # Normalisasi standar ImageNet
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Instansiasi dataset kustom untuk client_id (partition_id) ini
    dataset = FederatedSkinLesionDataset(
        manifest_path=MANIFEST_FILE,
        images_dir=IMAGE_FOLDER,
        transform=pytorch_transforms,
    )

    # Bagi data klien: 80% Latih, 20% Validasi
    if len(dataset) == 0:
        # Jika klien tidak punya data, kembalikan loader kosong
        log(level=logging.WARNING, msg="Peringatan: Dataset klien kosong!")
        return DataLoader(dataset), DataLoader(dataset)

    # Handle jika klien hanya punya 1 gambar (tidak bisa dibagi)
    if len(dataset) < 5:  # Minimal 5 data untuk dibagi
        len_train = len(dataset)
        len_val = 0
        ds_train, ds_val = dataset, None  # Gunakan semua untuk train
        log(level=logging.WARNING, msg="Peringatan: Dataset klien sangat kecil!")
    else:
        len_train = int(len(dataset) * 0.8)
        len_val = len(dataset) - len_train
        ds_train, ds_val = random_split(
            dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42)
        )
        log(
            level=logging.INFO,
            msg=f"Data klien dibagi: {len_train} train / {len_val} val.",
        )

    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    if ds_val:
        testloader = DataLoader(ds_val, batch_size=batch_size)
        log(level=logging.INFO, msg="Loader validasi siap.")
    else:
        # Jika tidak ada data val, gunakan loader kosong
        testloader = DataLoader(torch.utils.data.TensorDataset(), batch_size=batch_size)
        log(
            level=logging.WARNING,
            msg="Peringatan: Tidak ada data validasi untuk klien ini.",
        )

    log(level=logging.INFO, msg="Data klien dimuat sepenuhnya.")
    return trainloader, testloader


def train(net: ResNet, trainloader, epochs, device):
    """Melatih model pada training set."""
    log(level=logging.INFO, msg="Memulai pelatihan lokal...")

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0

    if len(trainloader) == 0:
        print("Peringatan: Trainloader kosong, pelatihan dilewati.")
        log(
            level=logging.WARNING,
            msg="Peringatan: Trainloader kosong, pelatihan dilewati.",
        )
        return 0.0  # Kembalikan 0 loss

    log(level=logging.INFO, msg=f"Jumlah epoch: {epochs}")
    for epoch in range(epochs):
        log(level=logging.INFO, msg=f"Memulai epoch baru:{epoch + 1}")
        for images, labels in trainloader:
            log(level=logging.DEBUG, msg="Memproses batch baru...")
            # Pindahkan data ke device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            log(level=logging.DEBUG, msg=f"Batch loss: {loss.item():.4f}")

    avg_trainloss = running_loss / len(trainloader)

    log(level=logging.INFO, msg="Pelatihan lokal selesai.")

    return avg_trainloss


# =============================================================================
# 4. Fungsi Test (Dimodifikasi sedikit)
# =============================================================================
def test(net: ResNet, testloader, device):
    """Memvalidasi model pada test set."""
    log(level=logging.INFO, msg="Memulai evaluasi model...")

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    if len(testloader.dataset) == 0:
        log(
            level=logging.WARNING,
            msg="Peringatan: Testloader kosong, akurasi tidak dapat dihitung.",
        )
        return 0.0, 0.0  # Hindari pembagian dengan nol

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    log(level=logging.INFO, msg="Evaluasi model selesai.")
    return loss, accuracy


def get_weights(net: ResNet):
    """Mendapatkan weights model."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: ResNet, parameters):
    """Mengatur weights model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

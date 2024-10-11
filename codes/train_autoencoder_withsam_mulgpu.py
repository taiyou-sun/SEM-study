import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch.multiprocessing as mp

# SAMモデルの初期化
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
init_gpu = 6

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AnomalyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # SAMを使用してセグメンテーション
        mask = segment_with_sam(image.unsqueeze(0))
        
        return image, mask.squeeze(0)

# 画像の前処理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# SAMを使用したセグメンテーション
def segment_with_sam(image):
    # PyTorch テンソルを NumPy 配列に変換し、値を 0-255 の整数に変換
    image_np = (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    predictor.set_image(image_np)
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)
    return torch.from_numpy(masks[0]).float().unsqueeze(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# オートエンコーダの学習
def train_autoencoder(rank, world_size, model, train_loader, num_epochs):
    setup(rank, world_size)
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters())

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0
        for image, mask in train_loader:
            image, mask = image.to(rank), mask.to(rank)
            mask = mask.unsqueeze(1)
            input_tensor = torch.cat([image, mask], dim=1)
            
            optimizer.zero_grad()
            output = ddp_model(input_tensor)
            loss = criterion(output, input_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), 'autoencoder_model.pth')
        print("Model saved successfully.")
    
    cleanup()

# 異物検知
def detect_anomaly(model, image_path, threshold, device):
    model.eval()
    image = preprocess_image(image_path).to(device)
    mask = segment_with_sam(image).to(device)
    input_tensor = torch.cat([image, mask], dim=1)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    diff = torch.abs(output - input_tensor)
    anomaly_map = diff.mean(dim=1).squeeze().cpu().numpy()
    
    return anomaly_map > threshold

def get_files_in_directory(directory_path):
    # 指定されたディレクトリ内のファイル名を格納するリストを初期化
    files = []
    
    # ディレクトリ内のすべてのファイルとフォルダをリストアップ
    for item in os.listdir(directory_path):
        # アイテムへのフルパスを取得
        item_path = os.path.join(directory_path, item)
        
        # アイテムがファイルであれば、リストに追加
        if os.path.isfile(item_path):
            files.append(item_path)
    
    return files

# メイン処理
def main(rank, world_size):
    print(rank)
    
    # データセットとデータローダーの準備
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    directory_path = "/mnt/data/datasets/SEM/dataset_v1/normal/"
    sem_datasets = get_files_in_directory(directory_path)
    
    train_dataset = AnomalyDataset(sem_datasets, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

    model = AutoEncoder()
    train_autoencoder(rank, world_size, model, train_loader, num_epochs=20)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
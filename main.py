import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils.KITTIOdometryDataset import KITTIOdometryDataset


EMBED_DIM = 128
PATCH_SIZE = 16
MLP_NODES = 128
OUT_DIM = 12

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.conv(x)                  
        x = x.flatten(2).permute(0, 2, 1)  
        return x

class TransformEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, MLP_NODES),
            nn.GELU(),
            nn.Linear(MLP_NODES, MLP_NODES),
            nn.GELU(),
            nn.Linear(MLP_NODES, OUT_DIM)
        )

    def forward(self, x):
        img1 = x["img1"]
        img2 = x["img2"]

        z1 = self.encoder(img1)                  
        z2 = self.encoder(img2)                  

        z = torch.cat([z1, z2, z2 - z1], dim=-1)  
        z = z.mean(dim=1)                         

        return self.head(z)                      


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = KITTIOdometryDataset(root="/kaggle/input/datasets/hocop1/kitti-odometry/", sequence="00", image_folder="image_2", transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = TransformEstimator().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0

        for batch in loader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            pred = model({
                "img1": img1,
                "img2": img2
            })

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0

        for batch in loader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            target = batch["target"].to(device)

            pred = model({
                "img1": img1,
                "img2": img2
            })

            loss = criterion(pred, target)
            total_loss += loss.item()

        return total_loss / len(loader)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
if __name__ == "__main__":
    main()

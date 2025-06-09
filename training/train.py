import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from tqdm import tqdm

# Define the custom encoder
class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()
        # Downsampling layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        enc1 = self.conv1(x)  # 64 channels, 128x128
        enc2 = self.conv2(enc1)  # 128 channels, 64x64
        enc3 = self.conv3(enc2)  # 256 channels, 32x32
        enc4 = self.conv4(enc3)  # 512 channels, 16x16
        enc5 = self.conv5(enc4)  # 512 channels, 8x8
        return enc5, [enc1, enc2, enc3, enc4]

# Define the decoder (Pix2Pix style with skip connections, no batch norm)
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        # Upsampling layers
        self.upconv1 = self.upconv_block(in_channels, 512)  # 8x8 -> 16x16
        self.upconv2 = self.upconv_block(512 + 512, 256)   # 16x16 -> 32x32, concat with enc4 (512 channels)
        self.upconv3 = self.upconv_block(256 + 256, 128)   # 32x32 -> 64x64, concat with enc3 (256 channels)
        self.upconv4 = self.upconv_block(128 + 128, 64)    # 64x64 -> 128x128, concat with enc2 (128 channels)
        self.upconv5 = self.upconv_block(64 + 64, 32)      # 128x128 -> 256x256, concat with enc1 (64 channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Pix2Pix uses Tanh for output
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connections):
        enc1, enc2, enc3, enc4 = skip_connections
        x = self.upconv1(x)  # 8x8 -> 16x16
        x = self.upconv2(torch.cat([x, enc4], dim=1))  # Concat with enc4
        x = self.upconv3(torch.cat([x, enc3], dim=1))  # Concat with enc3
        x = self.upconv4(torch.cat([x, enc2], dim=1))  # Concat with enc2
        x = self.upconv5(torch.cat([x, enc1], dim=1))  # Concat with enc1
        x = self.final_conv(x)
        return x

# Define the Pix2Pix-like model
class Pix2PixUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2PixUNet, self).__init__()
        self.encoder = CustomEncoder()
        self.decoder = UNetDecoder(512, out_channels)  # Encoder outputs 512 channels

    def forward(self, x):
        enc, skip_connections = self.encoder(x)
        dec = self.decoder(enc, skip_connections)
        return dec

class BeardCleanShavenDataset(Dataset):
    def __init__(self, beard_dir, clean_shaven_dir, transform=None):
        self.clean_shaven_images = sorted(os.listdir(clean_shaven_dir))
        self.beard_dir = beard_dir
        self.clean_shaven_dir = clean_shaven_dir
        self.transform = transform

    def __len__(self):
        return len(self.clean_shaven_images)

    def __getitem__(self, idx):
        beard_image = Image.open(os.path.join(self.beard_dir, self.clean_shaven_images[idx])).convert('RGB')
        clean_shaven_image = Image.open(os.path.join(self.clean_shaven_dir, self.clean_shaven_images[idx])).convert('RGB')

        if self.transform:
            beard_image = self.transform(beard_image)
            clean_shaven_image = self.transform(clean_shaven_image)

        return beard_image, clean_shaven_image

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
output_dir = './output_images'

def get_dataloaders(beard_dir, clean_shaven_dir):
    dataset = BeardCleanShavenDataset(beard_dir, clean_shaven_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    return train_loader, val_loader

# PSNR function
def psnr(output, target, max_value=1.0):
    mse = F.mse_loss(output, target)
    return 20 * torch.log10(max_value / torch.sqrt(mse))

# SSIM function with Gaussian kernel
def gaussian_kernel(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    return kernel_2d

def ssim_loss(output, target, window_size=11, size_average=True, val_range=None):
    output = output.float()
    target = target.float()
    kernel = gaussian_kernel(window_size, 1.5)
    kernel = kernel.view(1, 1, window_size, window_size).to(output.device)
    kernel = kernel.expand(output.shape[1], 1, window_size, window_size)
    mu_x = F.conv2d(output, kernel, padding=window_size//2, groups=output.shape[1])
    mu_y = F.conv2d(target, kernel, padding=window_size//2, groups=output.shape[1])
    sigma_x = F.conv2d(output**2, kernel, padding=window_size//2, groups=output.shape[1]) - mu_x**2
    sigma_y = F.conv2d(target**2, kernel, padding=window_size//2, groups=target.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(output * target, kernel, padding=window_size//2, groups=output.shape[1]) - mu_x * mu_y
    C1 = 0.01**2
    C2 = 0.03**2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean() if size_average else ssim_map.sum()

def denormalize(tensor, mean, std):
    for i in range(tensor.size(1)):
        tensor[:, i, :, :] = tensor[:, i, :, :] * std[i] + mean[i]
    return tensor

# Save example outputs
def save_example_output(model, dataloader, epoch, device):
    model.eval()
    with torch.no_grad():
        sample_input, _ = next(iter(dataloader))
        sample_input = sample_input.to(device)
        output = model(sample_input)
        denormalized_input = denormalize(sample_input.cpu(), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        denormalized_output = denormalize(output.cpu(), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        denormalized_input = torch.clamp(denormalized_input, 0, 1)
        denormalized_output = torch.clamp(denormalized_output, 0, 1)
        save_image(denormalized_output, os.path.join(output_dir, f'epoch_{epoch}_output.png'))
        save_image(denormalized_input, os.path.join(output_dir, f'epoch_{epoch}_input.png'))

# Training and evaluation loop
def train_and_evaluate(beard_dir="dataset/dataset/BeardFaces", clean_shaved_dir="dataset/dataset/CleanFaces", num_epochs=200, checkpoint_dir='training/checkpoints', test_image="media/test_image.png", output_path='media/output_image.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2PixUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter()
    train_loader, val_loader = get_dataloaders(beard_dir=beard_dir, clean_shaven_dir=clean_shaved_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        for i, (beard_image, clean_shaven_image) in enumerate(tqdm(train_loader)):
            beard_image = beard_image.to(device)
            clean_shaven_image = clean_shaven_image.to(device)
            optimizer.zero_grad()
            output_image = model(beard_image)
            loss = criterion(output_image, clean_shaven_image)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_psnr += psnr(output_image, clean_shaven_image).item()
            train_ssim += ssim_loss(output_image, clean_shaven_image).item()
            del beard_image, clean_shaven_image, output_image
            torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train PSNR: {train_psnr/len(train_loader):.4f}, Train SSIM: {train_ssim/len(train_loader):.4f}")
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for i, (beard_image, clean_shaven_image) in enumerate(val_loader):
                beard_image = beard_image.to(device)
                clean_shaven_image = clean_shaven_image.to(device)
                output_image = model(beard_image)
                loss = criterion(output_image, clean_shaven_image)
                val_loss += loss.item()
                val_psnr += psnr(output_image, clean_shaven_image).item()
                val_ssim += ssim_loss(output_image, clean_shaven_image).item()
                del beard_image, clean_shaven_image, output_image
                torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val PSNR: {val_psnr/len(val_loader):.4f}, Val SSIM: {val_ssim/len(val_loader):.4f}")
        save_example_output(model, val_loader, epoch, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pth')
            torch.save(model.state_dict(), model_checkpoint_path)  # Fixed: state_to -> state_dict
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_image = Image.open(test_image).convert('RGB')  
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = output_tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(torch.clamp(output_tensor * 0.5 + 0.5, 0, 1))
    output_image.save(output_path)
    return output_image
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

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

def test(image_path, model_path, output_path='./output_image_test.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2PixUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = output_tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(torch.clamp(output_tensor * 0.5 + 0.5, 0, 1))
    output_image.save(output_path)
    return output_image
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """A helper module with two convolutional layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Args:
        in_channels (int): Number of channels in the input image (3 for RGB).
        out_classes (int): Number of output classes (e.g., 16 for DOTA).
    """
    def __init__(self, in_channels, out_classes):
        super().__init__()
        
        # --- Encoder (Down-sampling path) ---
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(512, 1024)

        # --- Decoder (Up-sampling path) ---
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512) # 512 (from upconv) + 512 (from skip)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256) # 256 + 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128) # 128 + 128
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64) # 64 + 64

        # --- Final Output Layer ---
        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(x4))
        
        # --- Decoder with Skip Connections ---
        u1 = self.upconv1(b)
        # Concatenate the up-sampled features with the skip connection from the encoder
        skip1 = torch.cat([u1, x4], dim=1) 
        u1_out = self.up1(skip1)
        
        u2 = self.upconv2(u1_out)
        skip2 = torch.cat([u2, x3], dim=1)
        u2_out = self.up2(skip2)
        
        u3 = self.upconv3(u2_out)
        skip3 = torch.cat([u3, x2], dim=1)
        u3_out = self.up3(skip3)
        
        u4 = self.upconv4(u3_out)
        skip4 = torch.cat([u4, x1], dim=1)
        u4_out = self.up4(skip4)
        
        return self.out_conv(u4_out)
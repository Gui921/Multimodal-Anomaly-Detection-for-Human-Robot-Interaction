import torch
import torch.nn as nn

class Tiny3DConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel=(1,3,3), padding=(0,1,1)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(1, out_ch)  # GroupNorm with 1 group = LayerNorm across channels
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class TinyVideoUNetAE(nn.Module):
    def __init__(self, in_channels=1, base_channels=4, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = Tiny3DConvBlock(in_channels, base_channels, stride=1)         # -> 4
        self.enc2 = Tiny3DConvBlock(base_channels, base_channels*2, stride=2)     # -> 8
        self.enc3 = Tiny3DConvBlock(base_channels*2, base_channels*4, stride=2)   # -> 16

        # Bottleneck
        self.bottleneck = Tiny3DConvBlock(base_channels*4, latent_dim, stride=2)  # -> 32 latent

        # Decoder
        self.up3 = nn.ConvTranspose3d(latent_dim, base_channels*4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dec3 = Tiny3DConvBlock(base_channels*4 + base_channels*4, base_channels*2)

        self.up2 = nn.ConvTranspose3d(base_channels*2, base_channels*2, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dec2 = Tiny3DConvBlock(base_channels*2 + base_channels*2, base_channels)

        self.up1 = nn.ConvTranspose3d(base_channels, base_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dec1 = Tiny3DConvBlock(base_channels + base_channels, base_channels)

        # Final reconstruction
        self.final = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)   # -> base
        e2 = self.enc2(e1)  # -> base*2
        e3 = self.enc3(e2)  # -> base*4

        b = self.bottleneck(e3)

        u3 = self.up3(b)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(d3)

        u2 = self.up2(d3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.final(d1))
        return out

class ShallowVideoUNetAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*2, latent_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU()
        )

        # Decoder (mirrors encoder)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )

        # Final reconstruction
        self.final = nn.Conv3d(base_channels*2, in_channels, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # Bottleneck
        b = self.bottleneck(e2)

        # Decoder + skips
        d2 = torch.cat([self.dec2(b), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)

        out = self.final(d1)
        return torch.sigmoid(out)

class VideoUNetAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, base_channels, 3, padding=1),
                                  nn.BatchNorm3d(base_channels), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*4), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv3d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*8), nn.ReLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv3d(base_channels*8, latent_dim, 3, stride=2, padding=1),
                                        nn.BatchNorm3d(latent_dim), nn.ReLU())

        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose3d(latent_dim, base_channels*8, 4, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*8), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose3d(base_channels*16, base_channels*4, 4, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*4), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose3d(base_channels*8, base_channels*2, 4, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels*2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(base_channels*4, base_channels, 4, stride=2, padding=1),
                                  nn.BatchNorm3d(base_channels), nn.ReLU())

        self.final = nn.Conv3d(base_channels*2, in_channels, 3, padding=1)

    def forward(self, x):
        e1, e2, e3, e4 = self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x))), self.enc4(self.enc3(self.enc2(self.enc1(x))))
        b = self.bottleneck(e4)
        d4 = torch.cat([self.dec4(b), e4], dim=1)
        d3 = torch.cat([self.dec3(d4), e3], dim=1)
        d2 = torch.cat([self.dec2(d3), e2], dim=1)
        d1 = torch.cat([self.dec1(d2), e1], dim=1)
        return torch.sigmoid(self.final(d1))
    
class VideoUNetAutoencoderNoSkips(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=128, p = 0.3):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(),
            nn.Dropout3d(p)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*8, latent_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Dropout3d(p)
        )

        # Decoder (no skip connections, so channel sizes shrink)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels*8, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*4, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.Dropout3d(p)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Dropout3d(p)
        )

        # Final reconstruction
        self.final = nn.Conv3d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder (no skip connections)
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        out = torch.sigmoid(self.final(d1))
        return out
    
class ShallowVideoAutoencoderNoSkips(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64):
        super().__init__()
        # Encoder (2 layers only)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*2, latent_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU()
        )

        # Decoder (2 layers only, no skip connections)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )

        # Final reconstruction
        self.final = nn.Conv3d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # Bottleneck
        b = self.bottleneck(e2)

        # Decoder
        d2 = self.dec2(b)
        d1 = self.dec1(d2)

        out = torch.sigmoid(self.final(d1))
        return out
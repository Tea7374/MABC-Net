from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, MaxPool3d, Module

from utils import Conv3d, Conv1d


class C3DVideoEncoder(Module):
    """
    Video encoder (E_v): Process video frames to extract features.
    Input:
        V: (B, C, T, H, W)
    Output:
        F_v: (B, C_f, T)
    """

    def __init__(self, n_features=(64, 96, 128, 128)):
        super().__init__()

        n_dim0, n_dim1, n_dim2, n_dim3 = n_features

        # (B, 3, 512, 96, 96) -> (B, 64, 512, 32, 32)
        self.block0 = Sequential(
            Conv3d(3, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim0, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 3, 3))
        )

        # (B, 64, 512, 32, 32) -> (B, 96, 512, 16, 16)
        self.block1 = Sequential(
            Conv3d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 96, 512, 16, 16) -> (B, 128, 512, 8, 8)
        self.block2 = Sequential(
            Conv3d(n_dim1, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim2, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

            # (B, 128, 512, 8, 8) -> (B, 128, 512, 2, 2) -> (B, 512, 512) -> (B, 256, 512)
        self.block3 = Sequential(
            Conv3d(n_dim2, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Conv3d(n_dim3, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Rearrange("b c t h w -> b (c h w) t"),
            Conv1d(n_dim3 * 4, 256, kernel_size=1, stride=1, build_activation=LeakyReLU)
        )

    def forward(self, video: Tensor) -> Tensor:
        x = self.block0(video)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

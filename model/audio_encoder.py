from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Module, Sequential, LeakyReLU, MaxPool2d

from utils import Conv2d


class CNNAudioEncoder(Module):
    """
    Audio encoder (E_a): Process log mel spectrogram to extract features.
    Input:
        A': (B, F_m, T_a)
    Output:
        E_a: (B, C_f, T)
    """

    def __init__(self, n_features=(32, 64, 64)):
        super().__init__()

        n_dim0, n_dim1, n_dim2 = n_features

        # (B, 64, 2048) -> (B, 1, 64, 2048) -> (B, 32, 32, 1024)
        self.block0 = Sequential(
            Rearrange("b c t -> b 1 c t"),
            Conv2d(1, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 32, 32, 1024) -> (B, 64, 16, 512)
        self.block1 = Sequential(
            Conv2d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv2d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 64, 16, 512) -> (B, 64, 4, 512) -> (B, 256, 512)
        self.block2 = Sequential(
            Conv2d(n_dim1, n_dim2, kernel_size=(2, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Conv2d(n_dim2, n_dim2, kernel_size=(3, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Rearrange("b f c t -> b (f c) t")
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.block0(audio)
        x = self.block1(x)
        x = self.block2(x)
        return x

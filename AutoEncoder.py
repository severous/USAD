from torch import nn
import torch
class Encoder(nn.Module):
    def __init__(self, input_size, latent_size=10, ngpu=0):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        # 定义Encoder
        self.__Encoder = nn.Sequential(
            nn.Linear(input_size,input_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//4, latent_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoder = self.__Encoder(x)
        return encoder


class Decoder(nn.Module):

    def __init__(self, input_size, latent_size=10, ngpu=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        # 定义Decoder
        self.__Decoder = nn.Sequential(
            nn.Linear(latent_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size),
            nn.Sigmoid()
        )

    def forward(self,x):
        decoder = self.__Decoder(x)

        return decoder












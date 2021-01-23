from AutoEncoder import Encoder, Decoder
import torch
from torch import nn

class USAD(nn.Module):
    def __init__(self,input_size, latent_size,ngpu):
        super(USAD,self).__init__()
        self.encoder = Encoder(input_size=input_size,latent_size=latent_size,ngpu=ngpu)
        self.decoder1 = Decoder(input_size=input_size,latent_size=latent_size,ngpu=ngpu)
        self.decoder2 = Decoder(input_size=input_size,latent_size=latent_size,ngpu=ngpu)

    def forward(self, input):

        Z = self.encoder(input)
        W1_prime = self.decoder1(Z)
        W2_prime = self.decoder2(Z)
        W2_double_prime = self.decoder2(self.encoder(W1_prime))

        return W1_prime, W2_prime, W2_double_prime
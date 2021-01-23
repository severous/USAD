import torch.optim as optim
import math
from util import Loss
import torch

class Trainer():
    def __init__(self, encoder,decoder1,decoder2, lrate, device, alpha, beta, step_size=10):
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2

        self.encoder.to(device)
        self.decoder1.to(device)
        self.decoder2.to(device)

        self.optimizer1 = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder1.parameters()}])
        self.optimizer2 = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder2.parameters()}])
        self.loss1 = Loss(mode="AE1")
        self.loss2 = Loss(mode="AE2")
        self.step = step_size
        self.alpha = alpha
        self.beta = beta



    def train(self, input, i):

        Z = self.encoder(input)
        W1_prime = self.decoder1(Z)
        W2_prime = self.decoder2(Z)
        W2_double_prime = self.decoder2(self.encoder(W1_prime))

        error1 = self.loss1(input, W1_prime, W2_double_prime, i+1)
        error2 = self.loss2(input, W2_prime, W2_double_prime, i+1)

        error1.backward(retain_graph=True)
        error2.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        return error1, error2

    def valid(self,input, i):
        Z = self.encoder(input)
        W1_prime = self.decoder1(Z)
        W2_prime = self.decoder2(Z)
        W2_double_prime = self.decoder2(self.encoder(W1_prime))

        error1 = self.loss1(input, W1_prime, W2_double_prime, i + 1)
        error2 = self.loss2(input, W2_prime, W2_double_prime, i + 1)

        return error1, error2

    def eval(self, input):
        AE1 = self.decoder1(self.encoder(input))
        AE2 = self.decoder2(self.encoder(AE1))

        reconstruct_error =self.alpha * torch.mean((input-AE1)**2, dim=1) +\
                           self.beta * torch.mean((input-AE2)**2, dim=1)

        return reconstruct_error






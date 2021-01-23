from AutoEncoder import Encoder, Decoder
import torch
from torch import nn
from torch import optim
import torch.utils.data as data_utils
import pickle
import numpy as np
from util import *
from trainer import Trainer
from Module import USAD
import numpy as np
import pandas as pd
from sklearn import preprocessing

###############
window_length = 12
latent_size = 33
batch_size = 128
lrate = 0.02
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
nepochs = 3000
min_valid_error = np.inf
dataset = "SWaT"
train_valid_split = 0.7
lambda_ = 1.6
train_loss1 = []
train_loss2 = []
valid_loss1 = []
valid_loss2 = []
y_hat = []
###############
##----------------------------------------SWAT--------------------------------##
train, test = get_data(dataset)
X_train_  = train[0]
X_test_ = test[0]
y_test_ = test[1]
X_train_ = preprocess(X_train_)
X_test_ = preprocess(X_test_)
X_train_ = X_train_[np.arange(window_length)[None, :] + np.arange(X_train_.shape[0]-window_length)[:, None]]
X_test = X_test_[np.arange(window_length)[None, :] + np.arange(X_test_.shape[0]-window_length)[:, None]]
y_test = y_test_[np.arange(window_length)[None, :] + np.arange(y_test_.shape[0]-window_length)[:, None]]

input_size=window_length * 51
latent_size= 40

X_train = X_train_[0:round(0.8 * len(X_train_))]
X_valid = X_train_[round(0.8 * len(X_train_)):]
train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(X_train).float().view(([X_train.shape[0], input_size]))
) , batch_size=batch_size, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(X_valid).float().view(X_valid.shape[0],input_size)
) , batch_size=batch_size, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(X_test).float().view(X_test.shape[0],input_size)
) , batch_size=batch_size, shuffle=False, num_workers=0)

tem = np.sum(y_test,axis=1)
y_test = np.where(tem>=1,1,0)

encoder = Encoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)
decoder1 = Decoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)
decoder2 = Decoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)

engine = Trainer(encoder=encoder, decoder1=decoder1,decoder2=decoder2,  lrate=lrate,device=device,
                 alpha=0.5, beta=0.5)
print("Training: Device->{}".format(device))
for epoch in range(nepochs):

    for step, (X_batch, ) in enumerate(train_loader):
        train_error1, train_error2 = engine.train(X_batch, epoch+1)
        train_loss1.append(train_error1.item())
        train_loss2.append(train_error2.item())

        if step % 10 ==0:
            log = "Epoch{:3d}->step{:4d}: LOSS-AE1: {:4f}\tLOSS-AE2: {:4f}"
            print(log.format(epoch, step, train_loss1[-1], train_loss2[-1]))

    for step, (X_batch_,) in enumerate(val_loader):
        valid_error1, valid_error2 = engine.train(X_batch_, epoch + 1)
        valid_loss1.append(valid_error1.item())
        valid_loss2.append(valid_error2.item())

    mtrain_loss1 = np.mean(train_loss1)
    mtrain_loss2 = np.mean(train_loss2)
    mvalid_loss1 = np.mean(valid_loss1)
    mvalid_loss2 = np.mean(valid_loss2)
    total_train_loss = mtrain_loss1 + mtrain_loss2
    total_valid_loss = mvalid_loss1 + mvalid_loss2
    log = "Epoch{:3d}:Train Loss1->{:4f}\tTrain Loss2 ->{:4f}\tTotal Train Loss->{:4f}\t" \
          "Valid Loss1->{:4f}\tValid Loss2->{:4f}\tTotal Valid Loss->{:4f}\t"

    print(log.format(epoch, mtrain_loss1, mtrain_loss2, total_train_loss, mvalid_loss1,
                     mvalid_loss2, total_valid_loss))

    for (X_test_batch, ) in test_loader:

        y_hat_batch = engine.eval(X_test_batch)
        y_hat.extend(y_hat_batch.tolist())
    y_hat = np.asarray(y_hat)
    threshold = ROC(y_test, y_hat)
    print(threshold)
    precision, recall, f1 = metrics(y_test, y_hat>threshold[0])
    y_hat = []
    log = "lambda:{:4f} epoch{:3d}: precison->{:4f} \trecall->{:4f} \tf1->{:4f}"
    print(log.format(threshold[0
                     ], epoch, precision, recall, f1))
    torch.save({"encoder":engine.encoder.state_dict(),
                "decoder1":engine.decoder1.state_dict(),
                "decoder2":engine.decoder2.state_dict()}, "./model/"+dataset + "_USAD.pth")


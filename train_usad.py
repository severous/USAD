from AutoEncoder import Encoder, Decoder
import torch
from torch import nn
from torch import optim
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
latent_size = 40
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

##-------------------------------------------------------########
train, test = get_data(dataset)
X_train_  = train[0]
X_train = X_train_[:round(len(X_train_) * train_valid_split)]
X_valid = X_train_[round(len(X_train_) * train_valid_split):]
X_test = test[0]
y_test = test[1]

y_test = test_window(y_test, window_length=window_length)
train_sliding_window = BatchSlidingWindow(
            array_size=len(X_train),
            window_size=window_length,
            batch_size=batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )

valid_sliding_window = BatchSlidingWindow(
            array_size=len(X_valid),
            window_size=window_length,
            batch_size=batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )

print(X_train.shape)
input_size = X_train.shape[1] * window_length
##----------------------------------------MSL--------------------------------##

encoder = Encoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)
decoder1 = Decoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)
decoder2 = Decoder(input_size=input_size, latent_size = latent_size,ngpu=ngpu)

engine = Trainer(encoder=encoder, decoder1=decoder1,decoder2=decoder2,  lrate=lrate,device=device,
                 alpha=0.5, beta=0.5)

# try:
#     checkpoint = torch.load("./model/"+dataset + "_USAD.pth")
#     engine.encoder.load_state_dict( checkpoint["encoder"])
#     engine.decoder1.load_state_dict(checkpoint["decoder1"])
#     engine.decoder2.load_state_dict(checkpoint["decoder2"])
#
# except:
#     print("----loading failed----")
#     pass
print("Training: Device->{}".format(device))
for epoch in range(nepochs):
    train_iterator = train_sliding_window.get_iterator([X_train])
    for step, (X_batch, ) in enumerate(train_iterator):
        nbatches = len(X_batch)
        x_batch_train = torch.as_tensor(X_batch.reshape((nbatches, input_size))).to(device)
        train_error1, train_error2 = engine.train(x_batch_train, epoch+1)
        train_loss1.append(train_error1.item())
        train_loss2.append(train_error2.item())

        if step % 10 ==0:
            log = "Epoch{:3d}->step{:4d}: LOSS-AE1: {:4f}\tLOSS-AE2: {:4f}"
            print(log.format(epoch, step, train_loss1[-1], train_loss2[-1]))
    valid_iterator = valid_sliding_window.get_iterator([X_valid])
    for step, (X_batch_,) in enumerate(valid_iterator):
        nbatches = len(X_batch_)
        x_batch_valid = torch.as_tensor(X_batch_.reshape(nbatches, input_size)).to(device)
        valid_error1, valid_error2 = engine.train(x_batch_valid, epoch + 1)
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

    for X_test_batch in test_batch(X_test, batch_size, window_length):
        nbatches = len(X_test_batch)
        x_batch_test = torch.as_tensor(X_test_batch.reshape((nbatches, input_size))).to(device)
        y_hat_batch = engine.eval(x_batch_test)
        y_hat.extend(y_hat_batch.tolist())
    y_hat = np.asarray(y_hat)
    threshold = ROC(y_test, y_hat)
    y_hat_adjust = adjust_predicts(y_hat, y_test,threshold[0])
    precision, recall, f1 = metrics(y_test, y_hat_adjust)

    y_hat = []
    log = "lambda:{:4f} epoch{:3d}: precison->{:4f} \trecall->{:4f} \tf1->{:4f}"
    print(log.format(threshold[0], epoch, precision, recall, f1))
    torch.save({"encoder":engine.encoder.state_dict(),
                    "decoder1":engine.decoder1.state_dict(),
                    "decoder2":engine.decoder2.state_dict()}, "./model/"+dataset + "_USAD.pth")


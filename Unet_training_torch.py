import os
import re
import scipy.misc
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import math
from sklearn.utils import shuffle
import scipy.signal as signal
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm._utils import _term_move_up
from tqdm import tqdm
from datetime import datetime
from quantize import QConv2d, RangeBN
from quantize_rl import  QLinear


ACT_FW = 0
ACT_BW = 0
GRAD_ACT_ERROR = 0
GRAD_ACT_GC = 0

MOMENTUM = 0.9

DWS_BITS = 8
DWS_GRAD_BITS = 16


def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=True, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)
def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = QLinear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m

def find_correlation(output, original):
        length = 360
        f = 500
        nseg = 30
        nover = 6
        shape = np.shape(output)
        correlations = np.zeros((12, shape[0]))
        for j in range(shape[0]):
            sample_r = output[j, :, :, :]
            sample_o = original[j, :, :, :]
            signals_r = np.zeros((12, length))
            signals_o = np.zeros((12, length))
            for i in range(int(shape[3]/2)):
                real_r = sample_r[:, :, 2*i]
                real_o = sample_o[:, :, 2 * i]
                imaginary_r = sample_r[:, :, 2*i+1]
                imaginary_o = sample_o[:, :, 2 * i + 1]
                complex_signal_r = real_r + 1j*imaginary_r
                complex_signal_o = real_o + 1j * imaginary_o
                _, X_r = signal.istft(complex_signal_r, f, window="hann", nperseg=nseg, noverlap=nover)
                _, X_o = signal.istft(complex_signal_o, f, window="hann", nperseg=nseg, noverlap=nover)
                correlations[i, j] = np.corrcoef(X_r, X_o, rowvar=False)[0, 1]
                signals_r[i, :] = np.array(X_r)
                signals_o[i, :] = np.array(X_o)
        avg_corr_channel = np.mean(correlations, axis=1)
        avg_corr_overall = np.mean(avg_corr_channel)
        out = avg_corr_overall
        return avg_corr_channel, out, X_r, X_o



now = datetime.now()
#conv bit width - fully connected layer bitwidth
bit_width=[0,0]
pruning_ratio=[0,0,0]
wandb.init(project='nih_project_2',name=str(pruning_ratio)+"bit_w"+str(bit_width)+str(now))
freq = 10
dim = 16
#learning_rate = 0.000001
learning_rate=1e-3
input_dim = 10
code = 64
data_normalize=False
d2 = np.load("/data1/nih_data/Data/EGM_images_P2-2.npy")
l2 = np.load("/data1/nih_data/Data/EKG_images_P2-2.npy")
d3 = np.load("/data1/nih_data/Data/EGM_images_P3-2.npy")
l3 = np.load("/data1/nih_data/Data/EKG_images_P3-2.npy")
d4 = np.load("/data1/nih_data/Data/EGM_images_P4-2.npy")
l4 = np.load("/data1/nih_data/Data/EKG_images_P4-2.npy")
d5 = np.load("/data1/nih_data/Data/EGM_images_P5-2.npy")
l5 = np.load("/data1/nih_data/Data/EKG_images_P5-2.npy")
d7 = np.load("/data1/nih_data/Data/EGM_images_P7-2.npy")
l7 = np.load("/data1/nih_data/Data/EKG_images_P7-2.npy")
d8 = np.load("/data1/nih_data/Data/EGM_images_P8-2.npy")
l8 = np.load("/data1/nih_data/Data/EKG_images_P8-2.npy")
d9 = np.load("/data1/nih_data/Data/EGM_images_P9-2.npy")
l9 = np.load("/data1/nih_data/Data/EKG_images_P9-2.npy")
d13 = np.load("/data1/nih_data/Data/EGM_images_P13-2.npy")
l13 = np.load("/data1/nih_data/Data/EKG_images_P13-2.npy")
d17 = np.load("/data1/nih_data/Data/EGM_images_P17-2.npy")
l17 = np.load("/data1/nih_data/Data/EKG_images_P17-2.npy")
d18 = np.load("/data1/nih_data/Data/EGM_images_P18-2.npy")
l18 = np.load("/data1/nih_data/Data/EKG_images_P18-2.npy")
d19 = np.load("/data1/nih_data/Data/EGM_images_P19-2.npy")
l19 = np.load("/data1/nih_data/Data/EKG_images_P19-2.npy")
d24 = np.load("/data1/nih_data/Data/EGM_images_P24-2.npy")
l24 = np.load("/data1/nih_data/Data/EKG_images_P24-2.npy")
d25 = np.load("/data1/nih_data/Data/EGM_images_P25-2.npy")
l25 = np.load("/data1/nih_data/Data/EKG_images_P25-2.npy")
d26 = np.load("/data1/nih_data/Data/EGM_images_P26-2.npy")
l26 = np.load("/data1/nih_data/Data/EKG_images_P26-2.npy")

data = d5
labels = l5
#data=np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26))
#labels = np.concatenate(())

#print(data.shape)
#print(labels.shape)
#exit()
data=np.transpose(data,(0,3,1,2)).astype(np.float32)
labels=np.transpose(labels,(0,3,1,2)).astype(np.float32)
#print(data.shape)
#print(labels.shape)
#exit()
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42)
print(type(data_train))
print(data_train.shape)
print(labels_train.shape)

shape_train = np.shape(data_train)
shape_test = np.shape(data_test)
print("Number of training samples: ", shape_train[0])
print("Number of testing samples: ", shape_test[0])
print("Dimension of Data:", shape_train, shape_test)
print("Number of all Data", shape_train[0]+shape_test[0])
num_train_examples = shape_train[0]
num_test_examples = shape_test[0]
batch_size = 16
batch_num = int(math.ceil(num_train_examples/batch_size))
print("Number of batches: ", batch_num)
n_epochs = 2000
save_steps = batch_num/1 * n_epochs  # Number of training batches between checkpoint saves


mean_train_x=np.mean(data_train[:,:],axis=0)
std_train_x=np.std(data_train[:,:],axis=0)
if data_normalize:
    data_train=(data_train-mean_train_x)/std_train_x

mean_train_y=np.mean(labels_train,axis=0)
std_train_y=np.std(labels_train,axis=0)
if data_normalize:
    labels_train=(labels_train-mean_train_y)/std_train_y
    
train_set=TensorDataset(torch.tensor(data_train),torch.tensor(labels_train))
train_loader=DataLoader(train_set,batch_size=int(batch_size),num_workers=0)


if data_normalize:
    data_test=(data_test-mean_train_x)/std_train_x
    #test_x[:,4].fill(1)
    labels_test=(labels_test-mean_train_y)/std_train_y

test_set=TensorDataset(torch.tensor(data_test),torch.tensor(labels_test))
test_loader=DataLoader(test_set,batch_size=int(batch_size),num_workers=0)

# data_test=torch.tensor(data_test)
# labels_test=torch.tensor(labels_test)
# data_test = Variable(data_test, volatile=True).cuda()
# labels_test = Variable(labels_test, volatile=True).cuda()



def conv2d(in_channel, out_channel, kshape, stride, padding):
    tmp_layers=[]
    #tmp_layers.append(nn.Conv2d(in_channel,out_channel, kshape, stride=stride, padding=padding))
    #tmp_layers.append(conv(in_channel,out_channel,kshape,stride=stride,padding=padding))
    #tmp_layers.append(nn.Tanh())
    #return nn.Sequential(*tmp_layers)
    return conv(in_channel,out_channel,kshape,stride=stride,padding=padding)
    
def deconv2d(in_channel, out_channel, kshape, stride, padding):
    tmp_layers=[]
    tmp_layers.append(nn.ConvTranspose2d(in_channel,out_channel, kshape, stride=stride, padding=padding))
    tmp_layers.append(nn.Tanh())
    return nn.Sequential(*tmp_layers)
    
def maxpool2d(ksize=2, stride=2):
    return nn.MaxPool2d(ksize, stride=stride)

def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor,mode='bilinear',align_corners=False)

def fullyConnected(input_features,output_features):
    tmp_layers=[]
    #tmp_layers.append(nn.Linear(input_features,output_features))
    #tmp_layers.append(Linear(input_features,output_features))
    #tmp_layers.append(nn.Tanh())
    #return nn.Sequential(*tmp_layers)
    return Linear(input_features,output_features)

def fullyConnected2(input_features,output_features):
    #return nn.Linear(input_features,output_features, bias=False)
    return Linear(input_features,output_features,bias=False)



def correlation(x, y):
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x-mx, y-my
    r_num = torch.mean(torch.mul(xm,ym))        
    r_den = torch.std(xm) * torch.std(ym)
    return -1 * r_num / r_den
    
def dropout(p):
    return nn.Dropout(p=p)
    
    
#def find_correlation(output, original):
#        length = 360
#        f = 500
#        nseg = 30
#        nover = 6
#        shape = np.shape(output)
#        correlations = np.zeros((12, shape[0]))
#        for j in range(shape[0]):
#            sample_r = output[j, :, :, :]
#            sample_o = original[j, :, :, :]
#            signals_r = np.zeros((12, length))
#            signals_o = np.zeros((12, length))
#            for i in range(int(shape[3]/2)):
#                real_r = sample_r[:, :, 2*i]
#                real_o = sample_o[:, :, 2 * i]
#                imaginary_r = sample_r[:, :, 2*i+1]
#                imaginary_o = sample_o[:, :, 2 * i + 1]
#                complex_signal_r = real_r + 1j*imaginary_r
#                complex_signal_o = real_o + 1j * imaginary_o
#                _, X_r = signal.istft(complex_signal_r, f, window="hann", nperseg=nseg, noverlap=nover)
#                _, X_o = signal.istft(complex_signal_o, f, window="hann", nperseg=nseg, noverlap=nover)
#                correlations[i, j] = np.corrcoef(X_r, X_o, rowvar=False)[0, 1]
#                signals_r[i, :] = np.array(X_r)
#                signals_o[i, :] = np.array(X_o)
#        avg_corr_channel = np.mean(correlations, axis=1)
#        avg_corr_overall = np.mean(avg_corr_channel)
#        out = avg_corr_overall
#        return avg_corr_channel, out


def batch_norm(num_features):
    return nn.BatchNorm2d(num_features,eps=0.001,momentum=0.01)
def batch_norm1d(num_features):
    return nn.BatchNorm1d(num_features,eps=0.001,momentum=0.01)

class egmnet(nn.Module):
    def __init__(self):
        super(egmnet, self).__init__()
        keep_prob1=0.1
        keep_prob2=0.5
        keep_prob3=0
        self.lc1=conv2d(input_dim,48,7,1,7//2)
        self.lb1=batch_norm(48)
        self.lp1=maxpool2d()
        self.ldo1=dropout(keep_prob1)
        self.lc2=conv2d(48,96,5,1,5//2)
        self.lb2=batch_norm(96)
        self.lp2=maxpool2d()
        self.ldo2=dropout(keep_prob1)
        #reshape
        self.lfc1=fullyConnected(dim//4*dim//4*96,dim//4*dim//4*60)
        self.lbfc1=batch_norm1d(dim//4*dim//4*60)
        self.ldo3=dropout(keep_prob2)
        self.lfc2=fullyConnected(dim//4*dim//4*60,dim//4*dim//4*40)
        self.lbfc2=batch_norm1d(dim//4*dim//4*40)
        self.ldo4=dropout(keep_prob2)
        self.lfc3=fullyConnected(dim//4*dim//4*40,code)  
        self.lbfc3=batch_norm1d(code)
        
        
        self.lfc4=fullyConnected(code, dim//4*dim//4*40)
        self.lbfc4=batch_norm1d(dim//4*dim//4*40)
        self.ldo5=dropout(keep_prob2)
        self.lfc5=fullyConnected(dim//4*dim//4*40,dim//4*dim//4*60)
        self.lbfc5=batch_norm1d(dim//4*dim//4*60)
        self.ldo6=dropout(keep_prob2)
        self.lfc6=fullyConnected(dim//4*dim//4*60,dim//4*dim//4*96)
        self.lbfc6=batch_norm1d(dim//4*dim//4*96)
        self.ldo7=dropout(keep_prob2)
        #reshape
        self.ldc1=deconv2d(96,48,5,1,5//2)
        self.lbdc1=batch_norm(48)
        self.lup1=upsample()
        self.ldo8=dropout(keep_prob1)
        self.ldc2=deconv2d(48,24,7,1,7//2)
        self.lbdc2=batch_norm(24)
        self.lup2=upsample()
        self.ldo9=dropout(keep_prob1)
        self.lfc7=fullyConnected2(dim*dim*24,dim*dim*24)
        self.loutput=dropout(keep_prob3)
        
        #acttivation
        self.tanh=nn.ReLU()

    def forward(self,x,num_bits=[8,8]):
        x=self.lc1(x,num_bits[0])
        x=self.lb1(x)
        x=self.lp1(x)
        x=self.ldo1(x)
        x=self.lc2(x,num_bits[0])
        x=self.tanh(x)
        x=self.lb2(x)
        x=self.lp2(x)
        x=self.ldo2(x)
        x=x.view(-1,dim//4*dim//4*96)
        x=self.lfc1(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc1(x)
        x=self.ldo3(x)
        x=self.lfc2(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc2(x)
        x=self.ldo4(x)
        x=self.lfc3(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc3(x)
        x=self.lfc4(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc4(x)
        x=self.ldo5(x)
        x=self.lfc5(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc5(x)
        x=self.ldo6(x)
        x=self.lfc6(x,num_bits[1])
        x=self.tanh(x)
        x=self.lbfc6(x)
        x=self.ldo7(x)
        x=x.view(-1,96,dim//4,dim//4)
        x=self.ldc1(x)
        x=self.lbdc1(x)
        x=self.lup1(x)
        x=self.ldo8(x)
        x=self.ldc2(x)
        x=self.lbdc2(x)
        x=self.lup2(x)
        x=self.ldo9(x)
        x=x.view(-1,dim*dim*24)
        x=self.lfc7(x,num_bits[1])
        x=self.loutput(x)
        return x


net=egmnet()
net = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=1e-3)
# #optimizer = torch.optim.RMSprop(net.parameters(), lr=3e-3, momentum=0.9)
# #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,lr_step, eta_min=1e-7) 
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,lr_decay)
train_loss_logged=0
pbar = tqdm(range(n_epochs))
border = "="*50
clear_border = _term_move_up() + "\r" + " "*len(border) + "\r"
steps=0
for epoch in pbar:  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs,num_bits=bit_width)
        #loss = criterion(outputs, labels)
        loss=correlation(outputs, labels.view(-1,dim*dim*24))
        #loss=error_loss(outputs, labels, 0,1)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if (i+1) % 5 == 0:
            pbar.write('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 5))
            train_loss_logged=running_loss
            running_loss = 0.0
            wandb.log({'Train/loss': train_loss_logged/5,'steps': steps})
    steps+=1
    
    
    #pruning conv
    percent=pruning_ratio[0]
    total = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    


    #pruning fc
    percent=pruning_ratio[1]
    total = 0
    for m in net.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Linear):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.Linear):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total Linear params: {}, Pruned Linear params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))


    #pruning deconv
    percent=pruning_ratio[2]
    total = 0
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            total += m.weight.data.numel()
    conv_weights = torch.zeros(total)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(net.modules()):
        if isinstance(m, nn.ConvTranspose2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total ConvTranspose2d params: {}, Pruned ConvTranspose2d params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))

    
    running_loss_test=0
    logged_reconstructed_correlation=[]
    for ti, tdata in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        tinputs, tlabels = tdata

        tinputs = Variable(tinputs, volatile=True).cuda()
        tlabels = Variable(tlabels, volatile=True).cuda()
        toutputs = net(tinputs,num_bits=bit_width)
        tloss=correlation(toutputs, tlabels.view(-1,dim*dim*24))
        (avg_corr_channel, out, X_r, X_o)=find_correlation(np.transpose(toutputs.view(-1,24,dim,dim).cpu().detach().numpy(),(0,2,3,1)).astype(np.float32),\
                         np.transpose(tlabels.cpu().detach().numpy(),(0,2,3,1)).astype(np.float32))
        logged_reconstructed_correlation.append(out)
        running_loss_test+=tloss.item()
        #allc, corr = find_correlation(toutputs.view(-1,24,dim,dim).cpu().data, tlabels.cpu().data)
        #print('averaged correlation: ', corr)
        np.save("saved_waves/X_r"+str(ti)+".npy", X_r)
        np.save("saved_waves/X_o"+str(ti)+".npy", X_o)
    print('Reconstructed correlation',sum(logged_reconstructed_correlation)/len(logged_reconstructed_correlation))
    print('test loss: ',running_loss_test/(ti+1))
    wandb.log({'Test/loss': running_loss_test/(ti+1), 'epochs': epoch})
    wandb.log({'Test/reccor': sum(logged_reconstructed_correlation)/len(logged_reconstructed_correlation), 'epochs': epoch})

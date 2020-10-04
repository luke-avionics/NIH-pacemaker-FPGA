import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import sys 
from tqdm import tqdm
from tqdm._utils import _term_move_up

torch.manual_seed(0)    
np.random.seed(0)   

M1=48
N1=2
C1=16
H1=22
K1=7

M2=96
N2=48
C2=8
H2=12
K2=5


M3=4*4*60
C3=1
H3=1
K3=1

N4=4*4*40

N5=64

def custom_fc(weight,bias,input0):
    output0=np.zeros([1,weight.shape[0],1,1])
    for out_ch in range(weight.shape[0]):
        for in_ch in range(weight.shape[1]):
            output0[0,out_ch,0,0]+=input0[0,in_ch,0,0]*weight[out_ch,in_ch,0,0]
    for out_ch in range(weight.shape[0]):
            output0[0,out_ch,0,0]=np.tanh(bias[out_ch]+output0[0,out_ch,0,0])
    return output0
def custom_deconv(weight, input0):
    return None
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d1=self.conv2d(N1,M1,K1)
        self.conv2d2=self.conv2d(N2,M2,K2)
        self.f1=self.conv2d(M2*C2//2*C2//2,M3,1)
        self.f2=self.conv2d(M3,N4,1)
        self.f3=self.conv2d(N4,N5,1)
        self.max1=nn.MaxPool2d(2, stride=2)
    def conv2d(self,input_channel, output_channel, k):
        conv2d_layers=[]
        conv2d_layers.append(nn.Conv2d(input_channel,output_channel,k, padding=k//2,bias=True))
        conv2d_layers.append(nn.Tanh())
        #conv2d_layers.append(nn.BatchNorm2d(output_channel))
        return nn.Sequential(*conv2d_layers)
    def forward(self, x):
        x =self.conv2d1(x)
        x =self.max1(x)
        x =self.conv2d2(x)
        x =self.max1(x)
        #print('input before')
        #print(x.data)
        x =x.view(1,M2*C2//2*C2//2,1,1)
        
#=================================================================================        


        #x =self.f1(x)
        #x =self.f2(x)
        #x =self.f3(x)
        return x
        
        
test_x=np.random.rand(1,N1,C1,C1)
test_x=test_x.astype(np.float32)
with open("test_x.inc",'w') as f:
    for i in range(N1):
        for j in range(C1):
            for k in range(C1):
                f.write(str(test_x[0][i][j][k])+"\n\r")

    
net1=Net()

# for name, param in net1.named_parameters():
    # if param.requires_grad:
        # print (name, param.data)
        

test_weight=net1.conv2d1[0].weight.data
test_bias=net1.conv2d1[0].bias.data
print(test_weight.shape)
test_weight2=net1.conv2d2[0].weight.data
test_bias2=net1.conv2d2[0].bias.data
print(test_weight2.shape)
test_weight3=net1.f1[0].weight.data
test_bias3=net1.f1[0].bias.data
print(test_weight3.shape)

test_weight4=net1.f2[0].weight.data
test_bias4=net1.f2[0].bias.data
print(test_weight4.shape)

test_weight5=net1.f3[0].weight.data
test_bias5=net1.f3[0].bias.data
print(test_weight5.shape)


with open("test_weight.inc","w") as f:
    for i in range(M1):
        for j in range(N1):
            for k in range(K1):
                for l in range(K1):
                    f.write(str(float(test_weight[i][j][k][l]))+"\n\r")   
    for i in range(M1):
        f.write(str(float(test_bias[i]))+"\n\r")


with open("test_weight2.inc","w") as f:
    for i in range(M2):
        for j in range(N2):
            for k in range(K2):
                for l in range(K2):
                    f.write(str(float(test_weight2[i][j][k][l]))+"\n\r")   
    for i in range(M2):
        f.write(str(float(test_bias2[i]))+"\n\r")
with open("test_weight3.inc","w") as f:
    for i in range(M3):
        for j in range(0,M2*C2//2*C2//2,2*C2//2*C2//2):
            for j1 in range(C2//2*C2//2):
                for k in range(1):
                    for l in range(1):
                        f.write(str(float(test_weight3[i][j+j1][k][l]))+"\n\r")
                        f.write(str(float(test_weight3[i][j+j1+C2//2*C2//2][k][l]))+"\n\r")   
    for i in range(M3):
        f.write(str(float(test_bias3[i]))+"\n\r")

with open("test_weight4.inc","w") as f:
    for i in range(N4):
        for j in range(M3):
            for k in range(1):
                for l in range(1):
                    f.write(str(float(test_weight4[i][j][k][l]))+"\n\r")
    for i in range(N4):
        f.write(str(float(test_bias4[i]))+"\n\r")


with open("test_weight5.inc","w") as f:
    for i in range(N5):
        for j in range(N4):
            for k in range(1):
                for l in range(1):
                    f.write(str(float(test_weight5[i][j][k][l]))+"\n\r")
    for i in range(N5):
        f.write(str(float(test_bias5[i]))+"\n\r")



        
tensor_test_x= torch.tensor(test_x)
tensor_output= net1(tensor_test_x)  
test_output=tensor_output.data

with open("test_output.inc","w") as f:
    for i in range(M2*C2//2*C2//2):
        for j in range(1):
            for k in range(1):
                f.write(str(float(test_output[0][i][j][k]))+"\n\r")   


#test_x=np.random.rand(1,M2*C2//2*C2//2,1,1).astype('float32')
#output0_from_torch=net1.f1(torch.tensor(test_x))
#output0=custom_fc(test_weight3,test_bias3,test_x)

#print(output0_from_torch.data[0,11,0,0])
#print(output0[0,11,0,0])

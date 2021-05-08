import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as dst
from torchvision.utils import save_image


class pingjunlayer(nn.Module):
    def __init__(self, k=0.2, dim=0):
        super(pingjunlayer, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, outputs):
        '''
        1: B*C*M*N的数据计算方差，得到B×1的方差
        2：方差和设定百分比得到分配总功率，并按照注水法分配功率p1，p2，p3....
        3: 产生均值由计算给定，方差为分配值的随机噪声，加入数据
        '''
        z_var = self.calculate(outputs)
        z_var = self.redistribute(z_var)#k给定
        #torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)dtype=outputs.dtype, device=outputs.device,
        noise = torch.randn_like(outputs,  requires_grad=False)
        z_var = z_var.cuda()
        outputs = outputs + noise*torch.sqrt(z_var)  #????

        return outputs

    def set(self, k):
        self.k = k

    def calculate(self,X):
        num = X.shape[self.dim]
        var = torch.zeros(num, device= X.device, requires_grad=False)

        for n in range(num):
            if self.dim == 0:
                var[n] = torch.std(X[n,:,:,:]).detach()
            else:
                var[n] = torch.std(X[:,n,:,:]).detach()
        var = torch.pow(var, 2)
        return var
 

    def redistribute(self, z_var):
        total_power = torch.sum(z_var * self.k)
        n = len(z_var)
        power=torch.ones(1, requires_grad=False) * (total_power/n)
        return power

class zhushuilayer(nn.Module):
    def __init__(self, k=0.2, dim=0):
        super(zhushuilayer, self).__init__()
        self.k = k
        self.dim = dim


    def forward(self, outputs):
        '''
        1: B*C*M*N的数据计算方差，得到B×1的方差
        2：方差和设定百分比得到分配总功率，并按照注水法分配功率p1，p2，p3....
        3: 产生均值由计算给定，方差为分配值的随机噪声，加入数据
        '''
        l,m,n,k = outputs.shape
        z_var = self.calculate(outputs)
        z_var = self.redistribute(z_var)#k给定
        #torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)dtype=outputs.dtype, device=outputs.device,
        noise = torch.zeros_like(outputs,  requires_grad=False)
        if self.dim == 0:
            for idx in range(l):
                sigma = torch.randn((m,n,k), device=noise.device)
                noise[idx,:,:,:] = torch.sqrt(z_var[idx]) * sigma
        else:
            for idx in range(m):
                sigma = torch.randn((l,n,k), device=noise.device)
                noise[:,idx,:,:] = torch.sqrt(z_var[idx]) * sigma
        

        outputs = outputs + noise  #????

        return outputs

    def set(self, k):
        self.k = k

    def calculate(self,X):
        num = X.shape[self.dim]
        var = torch.zeros(num, device= X.device, requires_grad=False)
        for n in range(num):
            if self.dim == 0:
                var[n] = torch.std(X[n,:,:,:]).detach()
            else:
                var[n] = torch.std(X[:,n,:,:]).detach()
        var = torch.pow(var, 2)
        return var
 

    def redistribute(self, z_var):
        total_power = torch.sum(z_var * self.k)
        # print(total_power)
        sorted, indices = torch.sort(z_var)
        power=torch.zeros_like(z_var,requires_grad=False)
        idx = 1

        while total_power>0:
            if sorted[idx]-sorted[idx-1]>0:
                dp = sorted[idx] -sorted[idx-1]
                if dp*idx < total_power:
                    total_power -= dp*idx
                else: 
                    dp = total_power / idx
                    total_power = 0
                
                for n in range(idx):
                    power[indices[n]] += dp

            idx += 1

            if idx>= len(z_var):
                dp = total_power / idx
                for n in range(idx):
                    power[indices[n]] += dp
                # print(torch.sum(power))
                # print(power)
                # print(power+z_var)
                break
        return power
        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # zhushui = zhushuilayer(0.05).to(device)
    # input = torch.randn((5,3,20,20),requires_grad= True).to(device)
    # output = zhushui(input)
    # print(output.shape)


    pingjun = pingjunlayer(0.05).to(device)
    input = torch.randn((5,3,20,20),requires_grad= True).to(device)
    output = pingjun(input)
    print(output.shape)

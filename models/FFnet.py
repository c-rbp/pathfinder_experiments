import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class FFConvNet(nn.Module):

    def __init__(self, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super().__init__()
        filter_size = filt_size
        padding_size = filter_size // 2
        
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, bias=False, padding=3)
        self.bn0 = nn.BatchNorm2d(25)
        # kernel = np.load("gabor_serre.npy")
        # self.conv0.weight.data = torch.FloatTensor(kernel)
        
        self.conv1 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn1 = nn.BatchNorm2d(25)
        
        self.conv2 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn2 = nn.BatchNorm2d(25)
        
        self.conv3 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn3 = nn.BatchNorm2d(25)
        
        self.conv4 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn4 = nn.BatchNorm2d(25)
        
        self.conv5 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn5 = nn.BatchNorm2d(25)

        self.conv6 = nn.Conv2d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn6 = nn.BatchNorm2d(25)

        self.conv7 = nn.Conv2d(25, 2, kernel_size=1)
        #self.bn7 = nn.BatchNorm2d(25)

    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out = self.conv0(x)
        out = self.bn0(out)
        #out = torch.pow(out, 2)
        #print(out.shape)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = F.relu(out)
        output = self.conv7(out)
        loss = criterion(output, target)
        jv_penalty = torch.tensor([1]).float().cuda()
        
        return output, jv_penalty, loss

import torch
from typing import Type
from torch import nn
from torchvision import transforms as transforms
from torch.nn import functional as F

def crop_add(skip:Type[torch.Tensor], target:Type[torch.Tensor])->torch.Tensor:
    cropped_skip = skip[:,:,(skip.shape[-2]-target.shape[-2])//2:(skip.shape[-2]+target.shape[-2])//2,(skip.shape[-1]-target.shape[-1])//2:(skip.shape[-1]+target.shape[-1])//2]
    return torch.cat((cropped_skip, target), dim=1)

class ConvBN(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, stride=1):
        super(ConvBN,self).__init__()
        self.conv = nn.Conv2d(Cin, Cin, kernel_size=kernel_size, stride=stride, groups=Cin, bias=False)
        self.batchnorm = nn.BatchNorm2d(Cin)
        self.pointwise = nn.Conv2d(Cin, Cout, kernel_size=1, padding=0, bias=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.pointwise(x)
        return x

class ConvN(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, stride=1):
        super(ConvN,self).__init__()
        self.conv = nn.Conv2d(Cin, Cin, kernel_size=kernel_size, stride=stride, groups=Cin, bias=False)
        #self.batchnorm = nn.BatchNorm2d(Cin)
        self.pointwise = nn.Conv2d(Cin, Cout, kernel_size=1, padding=0, bias=False)
    def forward(self,x):
        x = self.conv(x)
        #x = self.batchnorm(x)
        x = self.pointwise(x)
        return x

class UrbanGreenRegression(nn.Module):
    def __init__(self):
        super(UrbanGreenRegression, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(6,1,kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )

        self.fc_block_2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,7, False),
            nn.BatchNorm1d(7)
        )

    def forward(self,x):
        x = self.conv_block_1(x)
        #x = self.conv_block_2(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc_block_1(x)
        x = self.fc_block_2(x)
        return torch.softmax(x, dim=-1)

class Splitted_Regression(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(Splitted_Regression, self).__init__()
        self.device = device
        self.regression = UrbanGreenRegression() # 4차원 텐서만 받음.
        self.regression.load_state_dict(torch.load('/home/bcyoon/Byeongchan/Data/N12/Model/Segmentation/Regression/2022.7.26/Best_Model_Parameters_of_15:32_patch_10_pointwiseConv.pth'))
        self.regression.to(device)
        for param in self.regression.parameters():
            param.requires_grad = False
        #self.batchnorm = nn.BatchNorm2d(7)
    def forward(self, x):
        # x : (batch, 100, 6, 10, 10) 데이터
        out = torch.zeros(x.shape[0],7,100)
        out = out.to(device = self.device)
        for i in range(x.shape[0]):
            tmp = self.regression(x[i,:,:,:,:])
            #print(tmp.shape)
            out[i,:,:] = torch.transpose(tmp, 0, 1)
        out = out.view(x.shape[0], 7, 10, 10)
        #Bilinear Interpolation 추가할 것
        out = F.interpolate(out, size=(100,100), mode='bilinear')
        #out = self.batchnorm(out)
        return out

class SegBlock(nn.Module):
    def __init__(self, skip:bool, in_channel:int, out_channel:int, is_last:bool = False):
        super(SegBlock,self).__init__()
        self.skip = skip
        self.is_last = is_last
        self.conv1 = ConvBN(in_channel, out_channel, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBN(out_channel, out_channel, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class SegBlock_Batchnorm_Deactivated(nn.Module):
    def __init__(self, skip:bool, in_channel:int, out_channel:int, is_last:bool = False):
        super(SegBlock_Batchnorm_Deactivated,self).__init__()
        self.skip = skip
        self.is_last = is_last
        self.conv1 = ConvN(in_channel, out_channel, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvN(out_channel, out_channel, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channel:int = 3):
        super(UNet, self).__init__()
        self.enc1 = SegBlock(True, in_channel=in_channel,out_channel=64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.enc2 = SegBlock(True, 64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.enc3 = SegBlock(True, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)
        self.enc4 = SegBlock(True, 256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.1)
        self.dec0 = SegBlock(False, 512, 1024)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec1 = SegBlock(False,1024,512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dropout5 = nn.Dropout(0.1)
        self.dec2 = SegBlock(False,512,256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dropout6 = nn.Dropout(0.1)
        self.dec3 = SegBlock(False,256,128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dropout7 = nn.Dropout(0.1)
        self.dec4 = SegBlock(False,128,64, is_last = True)

    def forward(self,x):
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        x = self.dropout1(x)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        x = self.dropout2(x)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        x = self.dropout3(x)
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        x = self.dropout4(x)
        x = self.dec0(x)
        x = self.upconv1(x)
        x = self.dec1(crop_add(enc4, x))
        x = self.upconv2(x)
        x = self.dropout5(x)
        x = self.dec2(crop_add(enc3, x))
        x = self.upconv3(x)
        x = self.dropout6(x)
        x = self.dec3(crop_add(enc2, x))
        x = self.upconv4(x)
        x = self.dropout7(x)
        x = self.dec4(crop_add(enc1, x))
        return x

class UNet_Batchnorm_Deactivated(nn.Module):
    def __init__(self, in_channel:int = 3):
        super(UNet_Batchnorm_Deactivated, self).__init__()
        self.enc1 = SegBlock_Batchnorm_Deactivated(True, in_channel=in_channel,out_channel=64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.enc2 = SegBlock_Batchnorm_Deactivated(True, 64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.enc3 = SegBlock_Batchnorm_Deactivated(True, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)
        self.enc4 = SegBlock_Batchnorm_Deactivated(True, 256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.1)
        self.dec0 = SegBlock_Batchnorm_Deactivated(False, 512, 1024)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec1 = SegBlock_Batchnorm_Deactivated(False,1024,512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dropout5 = nn.Dropout(0.1)
        self.dec2 = SegBlock_Batchnorm_Deactivated(False,512,256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dropout6 = nn.Dropout(0.1)
        self.dec3 = SegBlock_Batchnorm_Deactivated(False,256,128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dropout7 = nn.Dropout(0.1)
        self.dec4 = SegBlock_Batchnorm_Deactivated(False,128,64, is_last = True)

    def forward(self,x):
        
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        x = self.dropout1(x)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        x = self.dropout2(x)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        x = self.dropout3(x)
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        x = self.dropout4(x)
        x = self.dec0(x)
        x = self.upconv1(x)
        x = self.dec1(crop_add(enc4, x))
        x = self.upconv2(x)
        x = self.dropout5(x)
        x = self.dec2(crop_add(enc3, x))
        x = self.upconv3(x)
        x = self.dropout6(x)
        x = self.dec3(crop_add(enc2, x))
        x = self.upconv4(x)
        x = self.dropout7(x)
        x = self.dec4(crop_add(enc1, x))
        return x

class UNet_Interpolation(nn.Module):
    def __init__(self, in_channel:int = 3):
        super(UNet_Interpolation, self).__init__()
        self.enc1 = SegBlock(True, in_channel=in_channel,out_channel=64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.enc2 = SegBlock(True, 64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.enc3 = SegBlock(True, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)
        self.enc4 = SegBlock(True, 256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.1)
        self.dec0 = SegBlock(False, 512, 1024)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec1 = SegBlock(False,1024,512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dropout5 = nn.Dropout(0.1)
        self.dec2 = SegBlock(False,512,256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dropout6 = nn.Dropout(0.1)
        self.dec3 = SegBlock(False,256,128)
        self.upconv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.dropout7 = nn.Dropout(0.1)
        self.dec4 = SegBlock(False,128,64, is_last = True)

    def forward(self,x):
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        x = self.dropout1(x)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        x = self.dropout2(x)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        x = self.dropout3(x)
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        x = self.dropout4(x)
        x = self.dec0(x)
        x = self.upconv1(x)
        x = self.dec1(crop_add(enc4, x))
        x = self.upconv2(x)
        x = self.dropout5(x)
        x = self.dec2(crop_add(enc3, x))
        x = self.upconv3(x)
        x = self.dropout6(x)
        x = self.dec3(crop_add(enc2, x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv4(x)
        x = self.dropout7(x)
        x = self.dec4(crop_add(enc1, x))
        return x

class UNet_Padding(nn.Module):
    def __init__(self, in_channel:int = 3):
        super(UNet_Padding, self).__init__()
        self.enc1 = SegBlock(True, in_channel=in_channel,out_channel=64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.enc2 = SegBlock(True, 64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.enc3 = SegBlock(True, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)
        self.enc4 = SegBlock(True, 256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.1)
        self.dec0 = SegBlock(False, 512, 1024)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec1 = SegBlock(False,1024,512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dropout5 = nn.Dropout(0.1)
        self.dec2 = SegBlock(False,512,256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dropout6 = nn.Dropout(0.1)
        self.dec3 = SegBlock(False,256,128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dropout7 = nn.Dropout(0.1)
        self.dec4 = SegBlock(False,128,64, is_last = True)

    def forward(self,x):
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#52
        enc1 = self.enc1(x)#48
        x = self.pool1(enc1)#24
        x = self.dropout1(x)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#28
        enc2 = self.enc2(x)#24
        x = self.pool2(enc2)#12
        x = self.dropout2(x)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#16
        enc3 = self.enc3(x)#12
        x = self.pool3(enc3)#6
        x = self.dropout3(x)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#10
        enc4 = self.enc4(x)#6
        x = self.pool4(enc4)#3
        x = self.dropout4(x)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#7
        x = self.dec0(x)#3
        x = self.upconv1(x)#6
        x = torch.cat((enc4,x), dim=1)#
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#10
        x = self.dec1(x)#6
        x = self.upconv2(x)#12
        x = self.dropout5(x)
        x = torch.cat((enc3, x), dim=1)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#16
        x = self.dec2(x)#12
        x = self.upconv3(x)#24
        x = self.dropout6(x)
        x = torch.cat((enc2, x), dim=1)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#28
        x = self.dec3(x)#24
        x = self.upconv4(x)#48
        x = self.dropout7(x)
        x = torch.cat((enc1, x), dim=1)
        x = F.pad(input=x, pad=(2,2,2,2), mode='reflect')#52
        x = self.dec4(x)#48
        return x
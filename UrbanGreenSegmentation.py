# %%
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Type
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pkgs import dataprepare
from pkgs import legacytraining
from pkgs import neuralnet


# %%
'''class UrbanGreenSegmentation(pl.LightningModule):
    def __init__(self, rotate_training_data : bool = False, train_ratio : float = 0.8, patch_size : int = 100, batch_size : int = 4, region:str = 'N12'):
        super(UrbanGreenSegmentation, self).__init__()
        raw_data_array, OHE_target_array, raw_target_array = dataprepare.prepare_raw_files(region)
        self.batch_size = batch_size
        self.Datasets = {
            'Train' : dataprepare.TrainDataset3(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, rotate = rotate_training_data, train_ratio = train_ratio),
            'Validation' : dataprepare.TrainDataset3(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, is_validating = True, rotate = rotate_training_data, train_ratio = train_ratio),
            'Prediction' : dataprepare.TrainDataset3(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, is_evaluating = True, train_ratio = train_ratio)
        }

        self.Dataloaders = {
            'Train' : DataLoader(self.Datasets['Train'], batch_size=batch_size),
            'Validation' : DataLoader(self.Datasets['Validation'], batch_size=batch_size),
            'Prediction' : DataLoader(self.Datasets['Prediction'], batch_size=batch_size)
        }
        
        # 3개 배치 사용시 메모리 5기가
        # 2개 배치 사용시 메모리 3.8기가

        self.unet = neuralnet.UNet()
        self.regression = neuralnet.Splitted_Regression()
        
        self.fc1 = nn.Conv2d(in_channels=64, out_channels=7)
        self.bn1 = nn.BatchNorm2d(7)
        self.bn2 = nn.BatchNorm2d(14)
        self.fc2 = nn.Conv2d(in_channels=14, out_channels=7)
        self.softmax = nn.Softmax2d()

    def forward(self, x_seg, x_reg):
        x_reg = self.regression(x_reg)
        x_seg = self.unet(x_seg)
        x_seg = self.fc1(x_seg)
        x_seg = self.bn1(x_seg)
        x_seg = torch.cat((x_reg, x_seg), dim=1)
        x_seg = self.bn2(x_seg)
        x_seg = self.fc2(x_seg)
        x_seg = self.softmax(x_seg)
        return x_seg


    def training_step(self, batch, batch_idx):
        x_seg, x_reg, y_seg, _ = batch
        y_hat = self(x_seg, x_reg)
        return {'loss' : F.cross_entropy(y_hat, y_seg)}

    def validation_step(self, batch, batch_idx):
        x_seg, x_reg, y_seg, _ = batch
        y_hat = self(x_seg, x_reg)
        return {
            'valid_loss' : F.cross_entropy(y_hat, y_seg),
            'y_hat' : y_hat.detach(),
            'y' : y_seg.detach()
        }

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(train_optimizer, T_max=10)
        return [train_optimizer], [train_scheduler]

    def train_dataloader(self):
        return DataLoader(self.Datasets['Train'], batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.Datasets['Validation'], batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.Datasets['Prediction'], batch_size = self.batch_size)'''

# %%
class UrbanGreenSegmentation(nn.Module):
    def __init__(self, in_channel:int=6):
        super(UrbanGreenSegmentation, self).__init__()
        
        # 3개 배치 사용시 메모리 5기가
        # 2개 배치 사용시 메모리 3.8기가

        self.unet = neuralnet.UNet(in_channel=in_channel)
        #self.regression = neuralnet.Splitted_Regression()
        
        self.fc1 = nn.Conv2d(in_channels=64, out_channels=7, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(7)
        #self.bn2 = nn.BatchNorm2d(14)
        #self.fc2 = nn.Conv2d(in_channels=14, out_channels=7, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x_seg):
        #x_reg = self.regression(x_reg)
        x_seg = self.unet(x_seg)
        x_seg = self.fc1(x_seg)
        x_seg = self.bn1(x_seg)
        #x_seg = torch.cat((x_reg, x_seg), dim=1)
        #x_seg = self.bn2(x_seg)
        #x_seg = self.fc2(x_seg)
        x_seg = self.softmax(x_seg)
        return x_seg


# %%
def main():
    # --- GPU selection --- #
    gpus = 6 # slot number (e.g., 3), no gpu use -> write just ' '
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    raw_data_array_N12 ,raw_target_array_N12, OHE_target_array_N12 = dataprepare.prepare_raw_files('N12')
    raw_data_array_H19 ,raw_target_array_H19, OHE_target_array_H19 = dataprepare.prepare_raw_files('H19')
    raw_data_array_M18 ,raw_target_array_M18, OHE_target_array_M18 = dataprepare.prepare_raw_files('M18')

    raw_data_array = np.concatenate((raw_data_array_N12, raw_data_array_H19, raw_data_array_M18), axis=-1)
    raw_target_array = np.concatenate((raw_target_array_N12, raw_target_array_H19, raw_target_array_M18), axis=-1)
    OHE_target_array = np.concatenate((OHE_target_array_N12, OHE_target_array_H19, OHE_target_array_M18), axis=-1)



    batch_size = 4
    patch_size = 100
    train_ratio = 0.8
    rotate_training_data = False
    Datasets_ver3 = {
        'Train' : dataprepare.TrainDataset4(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, rotate = rotate_training_data, train_ratio = train_ratio),
        'Validation' : dataprepare.TrainDataset4(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, is_validating = True, rotate = rotate_training_data, train_ratio = train_ratio),
        'Prediction' : dataprepare.TrainDataset4(raw_data_array, OHE_target_array, raw_target_array, patch_size = patch_size, is_evaluating = True, train_ratio = train_ratio)
    }
    Dataloaders_ver3 = {
        'Train' : DataLoader(Datasets_ver3['Train'], batch_size=batch_size),
        'Validation' : DataLoader(Datasets_ver3['Validation'], batch_size=batch_size),
        'Prediction' : DataLoader(Datasets_ver3['Prediction'], batch_size=2400//patch_size)
    }
    model = UrbanGreenSegmentation()
    criterion3 = nn.CrossEntropyLoss()
    #옵티마이저 바꿔보기
    optimizer3 = torch.optim.Adam(model.parameters(), lr=0.005)
    #스케줄러 steplr로 바꿔서 해보기. 
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size = 50, gamma=0.9)

    best_model_path = legacytraining.train_model(model, dataloaders=Dataloaders_ver3, criterion=criterion3, num_epochs = 100, optimizer=optimizer3, scheduler=scheduler3, path='../Data/N12/Model/Segmentation/', description='lr0.005', device=device)
    

# %%
main()



import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Type
import rasterio
from torchvision import transforms as transforms
import os
from torch.nn import functional as F
import tqdm

def minmax(array : Type[np.ndarray], dim = 0):
    min = np.min(array, axis=dim)
    max = np.max(array, axis=dim)
    array = (array-min)/(max-min)
    return array

def log_minmax(array : Type[np.ndarray]):
    min = array.min()
    array = array - min + 1
    array = np.log(array)
    max = array.max()
    array = (array)/(max)
    return array





def prepare_raw_files(region:str, categories = 7):
    if (os.path.exists(f'../Data/{region}/np/train_array.npy') and os.path.exists(f'../Data/{region}/np/target_array_OHE.npy')) and (os.path.exists(f'../Data/{region}/np/target_array_RAW.npy')):
        print('Preexisting Data Found. Load from it?')
        if 'no' == 'yes':
            train_array = np.load(f'../Data/{region}/np/train_array.npy')
            target_array_OHE = np.load(f'../Data/{region}/np/target_array_OHE.npy')
            target_array = np.load(f'../Data/{region}/np/target_array_RAW.npy')
        else:
            print('No Data Found. Loading from Raw Data')
            lidar_image = rasterio.open(f'../Data/{region}/{region}_lidar.tif').read()
            lidar_array = np.array(lidar_image)
            lidar_array = log_minmax(lidar_array)

            lidar_1n_image = rasterio.open(f'../Data/{region}/{region}_lidar_1n.tif').read()
            lidar_1n_array = np.array(lidar_1n_image)
            lidar_1n_array = log_minmax(lidar_1n_array)

            lidar_nt_image = rasterio.open(f'../Data/{region}/{region}_lidar_nt.tif').read()
            lidar_nt_array = np.array(lidar_nt_image)
            lidar_nt_array = log_minmax(lidar_nt_array)

            RGB2020_image = rasterio.open(f'../Data/{region}/{region}_RGB2020.tif').read()
            RGB2020_array = np.array(RGB2020_image)

            train_array = np.stack([lidar_array, lidar_1n_array, lidar_nt_array]).squeeze()
            train_array = np.concatenate((train_array,RGB2020_array))
            target_image = rasterio.open(f'../Data/{region}/{region}_newlc.tif').read()
            target_array = np.array(target_image, dtype=int).squeeze()
            if target_array.shape[0] > 7:
                print('Not fitting. Trimming Data.')
                target_array = np.where(target_array == 3, 9, target_array)
                target_array = np.where(target_array == 4, 9, target_array)
                target_array = np.where(target_array == 5, 11, target_array)
                target_array = np.where(target_array == 6, 9, target_array)
            target_array = np.where(target_array == 1, 0, target_array)
            target_array = np.where(target_array == 2, 1, target_array)
            target_array = np.where(target_array == 7, 2, target_array)
            target_array = np.where(target_array == 8, 3, target_array)
            target_array = np.where(target_array == 9, 4, target_array)
            target_array = np.where(target_array == 10, 5, target_array)
            target_array = np.where(target_array == 11, 6, target_array)

            target_array_OHE = np.zeros(shape=(7,2400,2400))
            num = np.unique(target_array)

            num = max(num.shape[0],7)
            encoded_target_array = np.eye(num)[target_array]
            for i in range(encoded_target_array.shape[-1]):
                target_array_OHE[i,:,:]=encoded_target_array[:,:,i]
            
    else:
        print('No Data Found. Loading from Raw Data')
        lidar_image = rasterio.open(f'../Data/{region}/{region}_lidar.tif').read()
        lidar_array = np.array(lidar_image)
        lidar_array = log_minmax(lidar_array)

        lidar_1n_image = rasterio.open(f'../Data/{region}/{region}_lidar_1n.tif').read()
        lidar_1n_array = np.array(lidar_1n_image)
        lidar_1n_array = log_minmax(lidar_1n_array)

        lidar_nt_image = rasterio.open(f'../Data/{region}/{region}_lidar_nt.tif').read()
        lidar_nt_array = np.array(lidar_nt_image)
        lidar_nt_array = log_minmax(lidar_nt_array)

        RGB2020_image = rasterio.open(f'../Data/{region}/{region}_RGB2020.tif').read()
        RGB2020_array = np.array(RGB2020_image)

        train_array = np.stack([lidar_array, lidar_1n_array, lidar_nt_array]).squeeze()
        train_array = np.concatenate((train_array,RGB2020_array))
        target_image = rasterio.open(f'../Data/{region}/{region}_newlc.tif').read()
        target_array = np.array(target_image, dtype=int).squeeze()
        if target_array.shape[0] > 7:
            print('Not fitting. Trimming Data.')
            target_array = np.where(target_array == 3, 9, target_array)
            target_array = np.where(target_array == 4, 9, target_array)
            target_array = np.where(target_array == 5, 11, target_array)
            target_array = np.where(target_array == 6, 9, target_array)
        target_array = np.where(target_array == 1, 0, target_array)
        target_array = np.where(target_array == 2, 1, target_array)
        target_array = np.where(target_array == 7, 2, target_array)
        target_array = np.where(target_array == 8, 3, target_array)
        target_array = np.where(target_array == 9, 4, target_array)
        target_array = np.where(target_array == 10, 5, target_array)
        target_array = np.where(target_array == 11, 6, target_array)

        target_array_OHE = np.zeros(shape=(7,2400,2400))
        num = np.unique(target_array)

        num = max(num.shape[0],7)
        encoded_target_array = np.eye(num)[target_array]
        for i in range(encoded_target_array.shape[-1]):
            target_array_OHE[i,:,:]=encoded_target_array[:,:,i]

    if categories == 5:
        #5 카테고리 분류
        # 0 : Building
        # 1 : The others
        # 2 : Grass
        # 3 : Shrub
        # 4 : Tree
        target_array = np.where(target_array == 2, 1, target_array)
        target_array = np.where(target_array == 3, 1, target_array)
        target_array = np.where(target_array == 4, 2, target_array)
        target_array = np.where(target_array == 5, 3, target_array)
        target_array = np.where(target_array == 6, 4, target_array)
        
        
    os.makedirs(f'../Data/{region}/np', exist_ok=True)
    np.save(f'../Data/{region}/np/train_array.npy', train_array)
    np.save(f'../Data/{region}/np/target_array_RAW.npy', target_array)
    np.save(f'../Data/{region}/np/target_array_OHE.npy', target_array_OHE)

    return train_array, target_array.astype(int), target_array_OHE.astype(int)

class TrainDataset2(Dataset):
    def __init__(self, data_array : Type[np.ndarray], target_array_OHE : Type[np.ndarray], target_array_RAW : Type[np.ndarray], patch_size : int, is_evaluating : bool = False, is_validating : bool = False, rotate : bool = False, train_ratio : float = 0.8):
        self.is_validating = is_validating
        self.is_evaluating = is_evaluating
        seed = 386579

        #print(f'Data shape: {data_array.shape} | Target shape: {target_array.shape}')

        self.data = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), data_array.shape[0], patch_size, patch_size))

        for i in range(0,data_array.shape[1]//patch_size):
            for j in range(0,data_array.shape[2]//patch_size):
                self.data[data_array.shape[1]//patch_size*i+j,:,:,:] = data_array[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]


        self.label_OHE = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), target_array_OHE.shape[0] ,patch_size, patch_size), dtype=float)
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_OHE[data_array.shape[1]//patch_size*k+l,:,:,:] = target_array_OHE[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

        self.label_RAW = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size),data_array.shape[0]+1))
        
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_RAW[data_array.shape[1]//patch_size*k+l,:] = np.bincount(target_array_RAW[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size].reshape(-1), minlength=7)/(patch_size*patch_size)


        if not is_evaluating:
            if rotate:
                for i in range(2):
                    rotated_data = np.rot90(self.data, k=i+1, axes=(-2, -1))
                    self.data = np.concatenate((self.data, rotated_data), axis=0)
                    rotated_label_OHE = np.rot90(self.label_OHE, k=i+1, axes=(-2, -1))
                    rotated_label_RAW = self.label_RAW
                    self.label_OHE = np.concatenate((self.label_OHE, rotated_label_OHE), axis=0)
                    self.label_RAW = np.concatenate((self.label_RAW, rotated_label_RAW), axis=0)

        train_size = int(self.data.shape[0]*train_ratio)
        index_array = np.random.RandomState(seed=seed).permutation(self.data.shape[0])
        self.train_index = index_array[0:train_size]
        self.valid_index = index_array[train_size:index_array.shape[0]]
        
        self.data = torch.as_tensor(self.data).float()
        self.label_OHE = torch.as_tensor(self.label_OHE).float()
        self.label_RAW = torch.as_tensor(self.label_RAW).float()

        self.data[:,3:6,:,:] = self.data[:,3:6,:,:]/255

    def __len__(self):
        if self.is_evaluating:
            return self.data.shape[0]

        if self.is_validating:
            return self.valid_index.shape[0]
        else:
            return self.train_index.shape[0]

    def __getitem__(self, idx):
        if self.is_evaluating:
            sample = torch.as_tensor(self.data[idx,:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[idx,:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[idx,:]).float()
            return sample, label_OHE, label_RAW
        
        if self.is_validating:
            sample = torch.as_tensor(self.data[self.valid_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.valid_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.valid_index[idx],:]).float()
        else:
            sample = torch.as_tensor(self.data[self.train_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.train_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.train_index[idx],:]).float()

        return sample, label_OHE, label_RAW

def mirror_extrapolate(inp:torch.Tensor, in_shape:tuple = (100,100), out_shape:tuple = (284,284)) -> torch.Tensor: # 배치 포함해서 input으로 넣어야 함
    pad = ((out_shape[0]-in_shape[0])//2,(out_shape[0]-in_shape[0])//2,(out_shape[0]-in_shape[0])//2,(out_shape[0]-in_shape[0])//2)
    return F.pad(inp, pad=pad, mode="reflect")

class TrainDataset3(Dataset):
    def __init__(self, data_array : Type[np.ndarray], target_array_OHE : Type[np.ndarray], target_array_RAW : Type[np.ndarray], patch_size : int, is_evaluating : bool = False, is_validating : bool = False, rotate : bool = False, train_ratio : float = 0.8):
        self.is_validating = is_validating
        self.is_evaluating = is_evaluating
        seed = 386579

        
        #print(f'Data shape: {data_array.shape} | Target shape: {target_array_OHE.shape}')

        self.data = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), data_array.shape[0], patch_size, patch_size))

        for i in range(0,data_array.shape[1]//patch_size):
            for j in range(0,data_array.shape[2]//patch_size):
                self.data[data_array.shape[1]//patch_size*i+j,:,:,:] = data_array[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

        '''self.label_OHE = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), target_array_OHE.shape[0] ,patch_size, patch_size), dtype=float)
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_OHE[data_array.shape[1]//patch_size*k+l,:,:,:] = target_array_OHE[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]'''

        self.label_OHE = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size),patch_size, patch_size), dtype=float)

        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_OHE[data_array.shape[1]//patch_size*k+l,:,:] = target_array_RAW[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size]

        self.label_RAW = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size),data_array.shape[0]+1))
        print(self.label_RAW.shape)
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_RAW[data_array.shape[1]//patch_size*k+l,:] = np.bincount(target_array_RAW[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size].reshape(-1), minlength=7)/(patch_size*patch_size)


        if not is_evaluating:
            if rotate:
                for i in range(2):
                    rotated_data = np.rot90(self.data, k=i+1, axes=(-2, -1))
                    self.data = np.concatenate((self.data, rotated_data), axis=0)
                    rotated_label_OHE = np.rot90(self.label_OHE, k=i+1, axes=(-2, -1))
                    rotated_label_RAW = self.label_RAW
                    self.label_OHE = np.concatenate((self.label_OHE, rotated_label_OHE), axis=0)
                    self.label_RAW = np.concatenate((self.label_RAW, rotated_label_RAW), axis=0)

        train_size = int(self.data.shape[0]*train_ratio)
        index_array = np.random.RandomState(seed=seed).permutation(self.data.shape[0])
        self.train_index = index_array[0:train_size]
        self.valid_index = index_array[train_size:index_array.shape[0]]
        
        self.data = torch.as_tensor(self.data).float()
        self.data[:,3:6,:,:] = self.data[:,3:6,:,:]/255
        self.data_seg = mirror_extrapolate(self.data).squeeze()[:,0:3,:,:]
        self.data_reg = torch.zeros((self.data.shape[0], 100, 6, self.data.shape[-2]//10, self.data.shape[-1]//10))
        print(self.data.shape)
        for k in tqdm.trange(self.data.shape[0]):
            for i in range(self.data.shape[-1]//10):
                for j in range(self.data.shape[-2]//10):
                    self.data_reg[k,10*i+j,:,:,:] = self.data[k,:,10*j:10*j+10, 10*i:10*i+10]


        self.label_OHE = torch.as_tensor(self.label_OHE).float()
        self.label_RAW = torch.as_tensor(self.label_RAW).float()


    def __len__(self):
        if self.is_evaluating:
            return self.data.shape[0]

        if self.is_validating:
            return self.valid_index.shape[0]
        else:
            return self.train_index.shape[0]

    def __getitem__(self, idx):
        if self.is_evaluating:
            data_reg = torch.as_tensor(self.data_reg[idx,:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[idx,:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[idx,:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[idx,:]).float()
            return data_seg, data_reg, label_OHE, label_RAW
        
        if self.is_validating:
            data_reg = torch.as_tensor(self.data_reg[self.valid_index[idx],:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[self.valid_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.valid_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.valid_index[idx],:]).float()
        else:
            data_reg = torch.as_tensor(self.data_reg[self.train_index[idx],:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[self.train_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.train_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.train_index[idx],:]).float()

        return data_seg, data_reg, label_OHE, label_RAW

class TrainDataset4(Dataset):
    def __init__(self, data_array : Type[np.ndarray], target_array_OHE : Type[np.ndarray], target_array_RAW : Type[np.ndarray], patch_size : int, is_evaluating : bool = False, is_validating : bool = False, rotate : bool = False, train_ratio : float = 0.8, categories = 7):
        self.is_validating = is_validating
        self.is_evaluating = is_evaluating
        seed = 386579

        
        #print(f'Data shape: {data_array.shape} | Target shape: {target_array_OHE.shape}')

        self.data = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), data_array.shape[0], patch_size, patch_size))
        print(f'Internal data array : {self.data.shape}')

        for i in range(0,data_array.shape[1]//patch_size):
            for j in range(0,data_array.shape[2]//patch_size):
                self.data[data_array.shape[2]//patch_size*i+j,:,:,:] = data_array[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                #print(f'Migrating data from {i*patch_size}:{(i+1)*patch_size},{j*patch_size}:{(j+1)*patch_size} to {data_array.shape[2]//patch_size*i+j}')

        '''self.label_OHE = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size), target_array_OHE.shape[0] ,patch_size, patch_size), dtype=float)
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_OHE[data_array.shape[1]//patch_size*k+l,:,:,:] = target_array_OHE[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]'''

        self.label_OHE = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size),patch_size, patch_size), dtype=float)

        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_OHE[data_array.shape[2]//patch_size*k+l,:,:] = target_array_RAW[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size]

        self.label_RAW = np.zeros(((data_array.shape[1]//patch_size) * (data_array.shape[2]//patch_size),categories))
        for k in range(0,data_array.shape[1]//patch_size):
            for l in range(0,data_array.shape[2]//patch_size):
                self.label_RAW[data_array.shape[2]//patch_size*k+l,:] = np.bincount(target_array_RAW[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size].reshape(-1), minlength=categories)/(patch_size*patch_size)


        if not is_evaluating:
            if rotate:
                for i in range(2):
                    rotated_data = np.rot90(self.data, k=i+1, axes=(-2, -1))
                    self.data = np.concatenate((self.data, rotated_data), axis=0)
                    rotated_label_OHE = np.rot90(self.label_OHE, k=i+1, axes=(-2, -1))
                    rotated_label_RAW = self.label_RAW
                    self.label_OHE = np.concatenate((self.label_OHE, rotated_label_OHE), axis=0)
                    self.label_RAW = np.concatenate((self.label_RAW, rotated_label_RAW), axis=0)

        train_size = int(self.data.shape[0]*train_ratio)
        index_array = np.random.RandomState(seed=seed).permutation(self.data.shape[0])
        self.train_index = index_array[0:train_size]
        self.valid_index = index_array[train_size:index_array.shape[0]]
        
        self.data = torch.as_tensor(self.data).float()
        self.data[:,3:6,:,:] = self.data[:,3:6,:,:]/255
        self.data_seg = mirror_extrapolate(self.data).squeeze()[:,:,:,:]
        self.data_reg = torch.zeros((self.data.shape[0], 100, 6, self.data.shape[-2]//10, self.data.shape[-1]//10))

        


        self.label_OHE = torch.as_tensor(self.label_OHE).float()
        self.label_RAW = torch.as_tensor(self.label_RAW).float()


    def __len__(self):
        if self.is_evaluating:
            return self.data.shape[0]

        if self.is_validating:
            return self.valid_index.shape[0]
        else:
            return self.train_index.shape[0]

    def __getitem__(self, idx):
        if self.is_evaluating:
            data_reg = torch.as_tensor(self.data_reg[idx,:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[idx,:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[idx,:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[idx,:]).float()
            return data_seg, data_reg, label_OHE, label_RAW
        
        if self.is_validating:
            data_reg = torch.as_tensor(self.data_reg[self.valid_index[idx],:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[self.valid_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.valid_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.valid_index[idx],:]).float()
        else:
            data_reg = torch.as_tensor(self.data_reg[self.train_index[idx],:,:,:]).float()
            data_seg = torch.as_tensor(self.data_seg[self.train_index[idx],:,:,:]).float()
            label_OHE = torch.as_tensor(self.label_OHE[self.train_index[idx],:]).float()
            label_RAW = torch.as_tensor(self.label_RAW[self.train_index[idx],:]).float()

        return data_seg, data_reg, label_OHE, label_RAW

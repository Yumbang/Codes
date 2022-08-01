import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Type
from torch import nn
from torch.optim import optimizer
import rasterio
import zipfile
from matplotlib import pyplot as plt
import datetime
from torchvision import transforms as transforms
import shutil
import torchmetrics
import os
import pytorch_lightning as pl
import sklearn
from torch.nn import functional as F
import tqdm

def save_result(model: Type[nn.Module], dataloader : Type[DataLoader], path:str, description:str = '', reference_data:str = '', patch_size:int = 60, now = datetime.datetime.now()):
    best_model = model
    os.makedirs(os.path.join(path,f'{now.year}.{now.month}.{now.day}/', f'{description}/','tmp/'), exist_ok=True)
    zipped_results = zipfile.ZipFile(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','RESULT_{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}.zip'), 'w')
    prediction = np.zeros((2400//patch_size,2400//patch_size,7))

    with tqdm.tqdm(enumerate(dataloader)) as data_pbar:
        data_pbar.set_description('Predicting...')
        for i, (data, index_OHE, index) in data_pbar:
            prediction[i, :, :] = best_model(data).detach().numpy()

    

    prediction_expanded = np.zeros((7,2400,2400))
    for i in range(2400//patch_size):
        for j in range(2400//patch_size):
            for k in range(7):
                prediction_expanded[k,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = prediction[i,j,k]

    reference_image = rasterio.open(reference_data)
    layer_index = [1,2,7,8,9,10,11]

    with tqdm.trange(prediction_expanded.shape[0]) as write_pbar:
        write_pbar.set_description('Writing data')
        for i in write_pbar:
            #print('a') 
            processed_tiff = rasterio.open(
                os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', 'tmp/', f'Result_{layer_index[i]}_{description}.tif'),
                'w',
                driver='GTiff',
                height=prediction_expanded.shape[1],
                width=prediction_expanded.shape[2],
                count=1,
                dtype=prediction_expanded.dtype,
                crs=reference_image.crs,
                transform=reference_image.transform,
            )
            #print('b')
            processed_tiff.write(prediction_expanded[i,:,:],1)
            processed_tiff.close()
            #print('c')
            zipped_results.write(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', 'tmp/', f'Result_{layer_index[i]}_{description}.tif'), f'Result_{layer_index[i]}_{description}.tif')

    zipped_results.close()
    return os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','RESULT_{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}.zip')

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=13, train_rate: float = 0.8, batch_size: int = 60, path:str = '../Data/N12/Model/', description:str = 'no_description', reference_data:str = ''): 
    train_loss_history = []
    valid_loss_history = []

    patch_size = dataloaders['Train'].dataset.data.shape[-1]
    training_patches = len(dataloaders['Train'].dataset)
    validating_patches = len(dataloaders['Validation'].dataset)
    print(f'Training Patches : {training_patches}\nValidating Patches : {validating_patches}')

    best_model_epoch = 0
    least_valid_loss = 100
    now = datetime.datetime.now()
    os.makedirs(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', 'tmp/'), exist_ok=True)
    zipped_model = zipfile.ZipFile(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', '{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}'+'.zip'), 'w')
    
    epoch_range = tqdm.trange(num_epochs)
    for epoch in epoch_range:

        train_running_loss = 0.0
        valid_running_loss = 0.0

        epoch_range.set_description(f'EPOCH #{epoch}')
        

        for state in ['Train', 'Validation']:
            #pbar = tqdm.tqdm(dataloaders[state])
            for inputs, labels_OHE, labels in dataloaders[state]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.to(device)
                
                outputs = model(inputs)
                
                optimizer.zero_grad()

                if state == 'Train':
                    model.train()
                    train_loss = criterion(outputs, labels)
                    train_loss.backward()
                    train_running_loss += train_loss.item() * inputs.size(0)
                    #pbar.set_description('Train')
                
                if state == 'Validation':
                    model.eval()
                    valid_loss = criterion(outputs, labels)
                    valid_running_loss += valid_loss.item() * inputs.size(0)
                    #pbar.set_description('Valid')

                optimizer.step()
                #valid_running_similarity += metric(outputs, labels)
                #print('validating')
                
                #print(f'{i}th batch')
            #pbar.clear()
            
        epoch_range.refresh()

        
        #print(f'Memory after a training : {torch.cuda.memory_allocated()/1024/1024}')

        epoch_train_loss = train_running_loss / training_patches
        epoch_valid_loss = valid_running_loss / validating_patches
        scheduler.step(epoch_valid_loss)

        #print(f'Valid loss: {epoch_valid_loss} | Train loss: {epoch_train_loss}')


        if epoch_valid_loss < least_valid_loss:
            least_valid_loss = epoch_valid_loss
            best_model_epoch = epoch

        train_loss_history.append(epoch_train_loss)      
        valid_loss_history.append(epoch_valid_loss)

        torch.save(model.state_dict(), os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/', '{0:0=2d}.pth'.format(epoch)))
        zipped_model.write(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/', '{0:0=2d}.pth'.format(epoch)))

    plt.figure(figsize=(20,8))
    plt.plot(train_loss_history, 'r-')
    plt.plot(valid_loss_history, 'bo')
    plt.savefig(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/', 'Tendency.png'), dpi=300)
    zipped_model.write(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/', 'Tendency.png'))
    zipped_model.writestr('README.txt', f'{description}\nThe best Model : #{best_model_epoch}th model with loss {least_valid_loss}\nOptimizer : {optimizer}\nLoss function : {criterion}\nBatch size : {batch_size}\nScheduler : {scheduler}\nPatch size : {patch_size}\nTotal epochs : {num_epochs}\nModel information :\n{model.modules}')
    
    print('Best loss: {:4f}, in Epoch #{:0=3d}'.format(least_valid_loss, best_model_epoch))    
    zipped_model.close()
    shutil.copy(src=os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', 'tmp/', '{0:0=2d}.pth'.format(epoch)), dst=os.path.join(path,f'{now.year}.{now.month}.{now.day}/', 'Best_Model_Parameters_of_{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}'+'.pth'))
    print('Model information is saved in '+os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/', '{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}'+'.zip'))

    model.load_state_dict(torch.load(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/', '{0:0=2d}.pth'.format(best_model_epoch))))
    result_path = save_result(model = model.to('cpu'), dataloader=dataloaders['Prediction'], path=path, description=description, reference_data=reference_data, patch_size=patch_size, now=now)
    print('Model result is saved in '+ result_path)
    
    shutil.rmtree(os.path.join(path,f'{now.year}.{now.month}.{now.day}/',f'{description}/','tmp/'))
    best_model_path = os.path.join(path,f'{now.year}.{now.month}.{now.day}/', 'Best_Model_Parameters_of_{0:0=2d}:{1:0=2d}'.format(now.hour, now.minute)+f'_{description}'+'.pth')
    return best_model_path
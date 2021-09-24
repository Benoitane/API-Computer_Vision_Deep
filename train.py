# Import packages
from typing import List, Any

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import glob
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timeit import default_timer as timer
import time
from utils_model import get_pretrained_model, path_and_comod, train
from utils_preprocessing import create_and_distribute_images, create_train_val_test_set, \
    checking_data_csv, description_cleanded_data
import matplotlib.pyplot as plt

np.random.seed(2021)
random.seed(2021)

def main():
    print('#### Beggining of code... Here is 2 questions (First time, you can answer n and n) ####')
    response_pre = str(input('Are preprocessing and assignment to train-val-test has been done (y/n) ?'))
    if response_pre == 'n':
        print('#### Preprocessing part ####')
        response = str(input('Do you want to define manually general paths for you specific computer (y/n) ?'))
        if response == 'n':
            path_csv = 'data/'
            print('path csv : {}'.format(path_csv))
            path_img = os.getcwd()+'/'
            print('path file : {}'.format(path_img))
            parent_dir = path_img + 'data/images/'
            print('path image : {}'.format(parent_dir))
        if response == 'y':
            path_csv = str(input('Path to the csv file ? (example : data/)'))
            path_img = str(input('Path to the folder where python files are ? (example : os.getcwd()+"/")'))
            parent_dir = path_img + 'data/images/'

        data = pd.read_csv(path_csv + 'data_set.csv')
        data_new = checking_data_csv(data, path_img)
        time.sleep(5)
        list_target_3, data_new_2, data_to_split, data_to_split_separately = description_cleanded_data(
                    data_new)
        time.sleep(5)
        train_data, val_data, test_data, l_train_t, l_val_t, l_test_t = create_train_val_test_set(data_new_2,
                                                                                                data_to_split,
                                                                                                data_to_split_separately,
                                                                                                list_target_3)
        time.sleep(5)
        create_and_distribute_images(parent_dir,
                                             train_data, val_data, test_data, l_train_t, l_val_t, l_test_t)
        time.sleep(5)


        print('#### End of Preprocessing part ####')

    if response_pre == 'y':
        print('#### Preprocessing already done ####')

    print('#### Training part ####')

    traindir, validdir, testdir, batch_size, save_file_name, checkpoint_path, train_on_gpu = path_and_comod(
        'mobilenet_v2', 128)
    print("Defining data augmentation")
    image_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }
    print("Uploading data")
    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
            datasets.ImageFolder(root=validdir, transform=image_transforms['val'])
    }
    print("Defining dataloader")
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    }

    files = n_output = 0
    for _, dirnames, filenames in os.walk(traindir):
        files += len(filenames)
        n_output += len(dirnames)

    model = get_pretrained_model('mobilenet_v2', n_output)
    print(model)

    print('Defining parameters : criterion, optimizer, epochs and scheduler')

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    epochs = 60
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=0.000001, verbose=True)
    print(save_file_name)
    print('#### Training is starting... ###')
    model, history = train(model, criterion, optimizer, scheduler, device, dataloaders['train'], dataloaders['val'],
                           save_file_name=save_file_name, max_epochs_stop=epochs / 3, n_epochs=epochs, print_every=2)

    print('#### Training is over... ###')

    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig("graphs/LOSS_EVO_TRAINING_"+str(batch_size)+"_"+str(epochs)+".png", bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig("graphs/ACC_EVO_TRAINING_"+str(batch_size)+"_"+str(epochs)+".png", bbox_inches="tight")
    print('#### plots are exported ###')

    print('#### Training is over and model is saved... ###')


if __name__ == '__main__':
    main()

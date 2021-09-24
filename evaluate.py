# Import packages
import random
import numpy as np
from utils_model import *
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from utils_model import evaluate, get_pretrained_model,path_and_comod,load_model

np.random.seed(2021)
random.seed(2021)


def main():
    print('Loading data')
    traindir, validdir, testdir, batch_size, save_file_name, checkpoint_path, train_on_gpu = path_and_comod(
        'mobilenet_v2', 128)
    image_transforms = {
        'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }
    data = {
        'test':
            datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }
    class_names = data['test'].classes
    dataloaders = {
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }
    print('Loading pretrained model...')
    model = load_model('mobilenet_v2-transfer.pt', len(class_names))
    criterion = nn.CrossEntropyLoss()

    print('Starting evaluation')
    evaluate(model, dataloaders['test'], criterion, class_names)
    print('Evaluation is over')

if __name__ == '__main__':
    main()

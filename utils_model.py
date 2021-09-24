import pandas as pd
import numpy as np
import random
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timeit import default_timer as timer
from PIL import Image
import matplotlib.pyplot as plt

random.seed(2042)


def path_and_comod(mod, bs):
    """
    Prepare paths and comodities for deep learning model

    Inputs
    --------
        mod (str): model you want to implement
        bs (int): batch size

    Outputs
    --------
        traindir (str): path to train label folders
        validdir (str): path to val label folders
        testdir (str): path to test label folders
        batch_size (int): batch size number
        save_file_name (str): name to save the model .pt
        checkpoint_path (str): name to save the model .pth
        train_on_gpu (str): 'cuda' if GPU is available else 'cpu'
    """
    traindir = 'data/images/train/'
    validdir = 'data/images/val/'
    testdir = 'data/images/test/'
    save_file_name = mod + '-' + 'transfer.pt'
    checkpoint_path = mod + '-' + 'transfer.pth'
    batch_size = bs

    train_on_gpu = cuda.is_available()
    print('Train on gpu: {}'.format(train_on_gpu))

    if train_on_gpu:
        print('{} gpus detected'.format(cuda.device_count()))

    return traindir, validdir, testdir, batch_size, save_file_name, checkpoint_path, train_on_gpu


def process_image(image_path):
    """
    Process an image path into a PyTorch tensor

    Inputs
    --------
        image_path (str): path to the image

    Outputs
    --------
        img_tensor (tensor): PyTorch tensor

    """
    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def get_pretrained_model(model_name, n_outputs):
    """
    Function to load pretrained model for transfer learning

    Inputs
    --------
        model_name (str): name of the model
        n_outputs (int): number of labels of classification problem

    Outputs
    --------
        model (PyTorch model): cnn to finetune

    """
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_outputs), nn.LogSoftmax(dim=1))

    if model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_outputs), nn.LogSoftmax(dim=1))

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_outputs), nn.LogSoftmax(dim=1))

    return model


def train(model, criterion, optimizer, scheduler, device, train_loader, valid_loader, save_file_name, max_epochs_stop,
          n_epochs, print_every):
    """Train a PyTorch Model

    Inputs
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        scheduler (PyTorch scheduler): change LR according to results during training session
        device (str): 'cuda' if GPU is available else 'cpu'
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Outputs
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy to plot
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for : {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # scheduler step
                scheduler.step(train_loss)
                # print('Scheduler step : Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

                if (epoch + 1) % print_every == 0:
                    print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    torch.save(model.state_dict(), 'save/' + save_file_name)
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')

                        model.load_state_dict(torch.load('save/' + save_file_name))
                        model.optimizer = optimizer

                        history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
                        return model, history

    model.optimizer = optimizer
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    return model, history


def load_model(save_file_name,numberoflabel):
    """Load a PyTorch model checkpoint

    Inputs
    --------
        save_file_name (str): saved model in '.pt'
        n_o (int): number of labels

    Outputs
    --------
        model (PyTorch model): cnn to train

    """

    model = get_pretrained_model(save_file_name.split('.')[0].split('-')[0], numberoflabel)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('save/'+ save_file_name))
    else:
        model.load_state_dict(torch.load('save/' + save_file_name, map_location=torch.device('cpu')))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def predict(image_path, model, class_name):
    """Make a prediction for an image using a trained model

    Inputs
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        class_name (list): list of labels

    Outputs
    --------
        predicted_class (str): predicted label of image
        real_class (str): real label if available in name
    """
    if len(image_path.split('/')) >= 2:
        real_class = image_path.split('/')[-2]
    else:
        real_class = 'no precision'
    img_tensor = process_image(image_path)
    # Resize
    if cuda.is_available():
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        output = model(img_tensor)
        _, index = output.topk(k=1, dim=1, largest=True, sorted=True)
        predicted_class = class_name[index[0]]
        return predicted_class, real_class


def evaluate(model, test_loader, criterion, class_name):
    """Measure the performance of a trained PyTorch model

    Inputs
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        criterion (PyTorch loss): criterion to minimize
        class_name (list): list of labels

    Outputs
    --------
        display summary of performances according to different classes on test set

    """

    test_loss = 0.0
    class_correct = list(0. for i in range(len(class_name)))
    class_total = list(0. for i in range(len(class_name)))
    model.eval()

    # iterate over test data
    for data, target in test_loader:

        batch_size = data.size(0)
        # print(batch_size)
        # move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(len(class_name)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                class_name[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_name[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Pytorch Tensor

    Inputs
    --------
        image (Pytorch tensor): image tensor

    Outputs
    --------
        ax (matplotlib axe): matplotlib.axes._subplots.AxesSubplot
        image (numpy.ndarray): tensor converted for displaying
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

def predict_top_k(image_path, model, class_name, topk):
    """Make a prediction for an image using a trained model

    Inputs
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        class_name (list): list of labels

    Outputs
    --------
        predicted_class (str): predicted label of image
        real_class (str): real label if available in name
    """
    if len(image_path.split('/')) >= 2:
        real_class = image_path.split('/')[-2]
    else:
        real_class = 'no precision'
    img_tensor = process_image(image_path)
    # Resize
    if cuda.is_available():
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            class_name[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class


def display_prediction_top_k(image_path, model, class_name, topk):
    """Display image and preditions from model

    Inputs
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        class_name (list): list of labels
        topk (int): number of classes you want to see

    Outputs
    --------
        No object returned. Plot exported in save/ folder
    """
    # Get predictions
    img, ps, classes, y_obs = predict_top_k(image_path, model, class_name, topk)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes)
    # Show the image
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    ax, img = imshow_tensor(img, ax=ax)
    # Set title to be the actual class
    ax.set_title(y_obs, size=20)
    ax = plt.subplot(1, 2, 2)
    # Plot a bar plot of predictions
    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.savefig("graphs/Top_"+str(topk)+"_prediction_for_"+str(image_path.split('/')[-1])+".png", bbox_inches="tight")

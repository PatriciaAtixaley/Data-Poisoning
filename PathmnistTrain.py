import os

import torch

import numpy as np

from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import PathMNIST, INFO

import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

from models.simple import SimpleNet

from models.resnet import ResNet18  # Import your custom ResNet18 model

 

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

 

cudnn.benchmark = True if use_cuda else False

 

writer = SummaryWriter(log_dir='./logs')

 

def get_model():

    model = ResNet18(num_classes=9, size=64)  # Adjust size if necessary

    model.to(device)

    return model

 

transform_train = transforms.Compose([

    transforms.RandomRotation(10),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale images

])

 

transform_val = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale images

])

 

train_dataset = PathMNIST(split='train', transform=transform_train, download=True)

valid_dataset = PathMNIST(split='val', transform=transform_val, download=True)

 

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

 

criterion = nn.CrossEntropyLoss()

 

# Initialize the model and optimizer

model = get_model()

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

 

# Path to save the trained model

save_path = './01_global_models/path/R1049.pt'

 

# Training function (unchanged)

def train(n_epochs, model, optimizer, criterion, scheduler, use_cuda, save_path):

    valid_loss_min = np.Inf

    patience = 5

    epochs_no_improve = 0

 

    for epoch in range(1, n_epochs + 1):

        train_loss = 0.0

        valid_loss = 0.0

        correct_train = 0

        total_train = 0

        correct_valid = 0

        total_valid = 0

 

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            if use_cuda:

                data, target = data.cuda(), target.cuda()

 

            target = target.view(-1)

 

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

 

            train_loss += (loss.item() - train_loss) / (batch_idx + 1)

 

            # Calculate training accuracy

            _, predicted = torch.max(output, 1)

            correct_train += (predicted == target).sum().item()

            total_train += target.size(0)

 

        # Validation Phase

        model.eval()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(valid_loader):

                if use_cuda:

                    data, target = data.cuda(), target.cuda()

                target = target.view(-1)

                output = model(data)

                loss = criterion(output, target)

                valid_loss += (loss.item() - valid_loss) / (batch_idx + 1)

                _, predicted = torch.max(output, 1)

                correct_valid += (predicted == target).sum().item()

                total_valid += target.size(0)

 

        writer.add_scalar('Training Loss', train_loss, epoch)

        writer.add_scalar('Validation Loss', valid_loss, epoch)

        writer.add_scalar('Training Accuracy', 100 * correct_train / total_train, epoch)

        writer.add_scalar('Validation Accuracy', 100 * correct_valid / total_valid, epoch)

 

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        print(f'Training Accuracy: {100 * correct_train / total_train:.2f}% \tValidation Accuracy: {100 * correct_valid / total_valid:.2f}%')

 

        # Save the model if validation loss improves

        if valid_loss < valid_loss_min:

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save({

                'epoch': epoch,

                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),

                'loss': valid_loss,

            }, save_path)

            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')

            valid_loss_min = valid_loss

            epochs_no_improve = 0

        else:

            epochs_no_improve += 1

        if epochs_no_improve >= patience:

            print("Early stopping triggered. Training stops.")

            break

        scheduler.step()

    return model

 

# Number of epochs to train for

n_epochs = 20  # Adjust this value based on your needs

trained_model = train(n_epochs, model, optimizer, criterion, scheduler, use_cuda, save_path)

 

writer.close()

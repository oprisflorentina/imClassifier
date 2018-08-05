import argparse
import torch
import numpy as np

from torch import nn, optim
from torch.autograd import Variable

from torchvision import datasets, models, transforms
from collections import OrderedDict


import time
import copy


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
         ]),
        'valid' : transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
         ]),
        'test' : transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
         ])
     }

    # Load the datasets with ImageFolder
    image_datasets = {
            'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
            'valid' : datasets.ImageFolder(train_dir, transform=data_transforms['valid']),
            'test'  : datasets.ImageFolder(train_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
            'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
            'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
            'test'  : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return image_datasets, dataloaders

# Build the network. Use a vgg network and create a new classifier
def build_network(hidden_layers, drop_p, model = 'vgg19'):
    if model == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif model == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("Please load a VGG network")
        return False

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Add the classifier
    # Started from the lesson, but I tried to adapt to a large variety of hidden layers
    classifier_data =  OrderedDict ()
    for i in range(len(hidden_layers)):
        classifier_data['dropout' + str(i+1)] = nn.Dropout(drop_p)

        if(i == 0):
            classifier_data['fc' + str(i+1)] = nn.Linear(25088, hidden_layers[i])
        elif(i < len(hidden_layers)):
            classifier_data['fc' + str(i+1)] = nn.Linear(hidden_layers[i-1], hidden_layers[i])
        else:
            classifier_data['fc' + str(i+1)] = nn.Linear(hidden_layers[i], 102)
            #classifier_data['fc' + str(i+1)] = nn.Linear(hidden_layers[i], 15)
        if(i < len(hidden_layers)):
            classifier_data['relu' + str(i+1)] = nn.ReLU()

    # add softmax layer
    classifier_data['output'] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(classifier_data)

    model.classifier = classifier
    return model

# Train the network
# The best model for training the network that I found was from PyTorch tutorials
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_network(dataloaders, image_datasets, hidden_layers, drop_p=0.25, learning_rate=0.001, epochs=10, model_network='vgg19', device='cpu', checkpoint='None'):

    # Setup dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

    # Setup start time - calculate the time
    start = time.time()

     # Load the model
    model = build_network(hidden_layers, drop_p, model_network)

    if model == False:
        raise ValueError('The model was not properly built. Make sure you use a proper VGG network. You model is {}'.format(model_network))

    # define the criterion and optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    # I noticed that SGD works better
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)
    print("Total Epochs {}: ".format(epochs))
    for e in range(epochs):
        # Each epoch has a training and validation phase
        print("Epoch {}: ".format(str(e + 1)))
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

             # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase.title(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save checkpoint
    if checkpoint is not 'None':
        save_checkpoint(image_datasets, model, model_network, hidden_layers, drop_p, checkpoint + '/checkpoint.pth')

    return model

# Save the checkpoint
# Folowed the example from this Class lessons
def save_checkpoint(image_datasets, model, model_network, hidden_layers, drop_p, savepath):
    model.class_to_idx = image_datasets['train'].class_to_idx
    # model = train_network(dataloaders, image_datasets, hidden_layers, 0.25, 0.001, 3, 'vgg19', device)
    checkpoint = {
                  'model_network' : model_network,
                  'class_to_idx': model.class_to_idx,
                  'hidden_layers': hidden_layers,
                  'drop_p': drop_p,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, savepath)


def main():

    # Parse arguments
    # train.py flowers_small --hidden_units 4096 512 --gpu=False --learning_rate=0.001 --drop_p=0.1 --epochs=4 --save_dir=checkpoints
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str)
    parser.add_argument('--save_dir', action="store", dest="save_dir")
    parser.add_argument('--arch', dest="arch", type=str, default="vgg19")

    parser.add_argument('--epochs', dest="epochs", type=int, default=13)
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.001)
    parser.add_argument('--hidden_units', nargs='+', type=int)
    parser.add_argument('--drop_p', dest="drop_p", type=float, default=0.25)

    parser.add_argument('--gpu', dest="gpu", default=False, type=bool)

    args = parser.parse_args()
    print(parser.parse_args())

    # setp devivce
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load the data
    image_datasets, dataloaders = load_data(args.data_directory)
    # Train the network
    model = train_network(dataloaders, image_datasets, args.hidden_units, args.drop_p, args.learning_rate, args.epochs, args.arch, device, args.save_dir)

    #print(image_datasets)
    #print(dataloaders)


if __name__== "__main__":
  main()

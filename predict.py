import torch
import numpy as np

from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image

import time
import argparse
import json

import train

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    # Resize image, must be (256, y) or (y, 256), where y > 256
    # https://stackoverflow.com/questions/41720557/image-thumbnail-with-at-least-a-certain-size-in-pil
    if pil_image.size[0] > pil_image.size[1]:
        new_width = 256 * pil_image.size[0] / pil_image.size[1]
        pil_image.thumbnail((new_width, 256), Image.ANTIALIAS) #constrain the height to be 256
    else:
        new_height = 256 * pil_image.size[1] / pil_image.size[0]
        pil_image.thumbnail((256, new_height), Image.ANTIALIAS) #otherwise constrain the width

    # Center Crop image
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = pil_image.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    # Convert the color channels
    np_image = np.array(pil_image)/255

    # Normalize data and reorder dimensions
    # https://stackoverflow.com/questions/13687256/is-it-right-to-normalize-data-and-or-weight-vectors-in-a-som
    np_image = np_image - np.array([0.485, 0.456, 0.406])
    np_image = np_image / np.array([0.229, 0.224, 0.225])
    np_image = np.ndarray.transpose(np_image, (2, 0, 1))

    return np_image

def predict(image_path, model, gpu=False, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    # Load and process image
    image = process_image(image_path)
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)

    # pass it through the model and get top-K
    probs, indices  = model(image).topk(topk)

    # Must apply softmax to probs
    # https://discuss.pytorch.org/t/any-way-to-get-confidence-for-class-predictions-from-variable/3960

    # Round percentages
    probs = torch.nn.functional.softmax(probs, dim=1)
    # I wanted the probabilities to show a readable number. I tried many combinations, followerd the errors and got to this formula
    probs = np.round(probs.detach().numpy()[0], 6)

    # Convert the indices to the actual class labels using class_to_idx
    # invert the dictionary so you get a mapping from index to class as well
    # https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
    class_to_idx = {val: key for key, val in model.class_to_idx.items()}

    labels = []
    for index in indices.numpy()[0]:
        labels.append(class_to_idx[index])

    return probs, labels

def main():
    # Parse arguments
    # python predict.py "flowers/test/1/image_06764.jpg" "checkpoints/checkpoint.pth" --top_k=6 --category_names="cat_to_name.json" --gpu=True
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', action="store", type=str)
    parser.add_argument('checkpoint', action="store", type=str)
    parser.add_argument('--top_k', dest="top_k", type=int)
    parser.add_argument('--category_names', dest="category_names", type=str)
    parser.add_argument('--gpu', dest="gpu", default=False, type=bool)
    parser.add_argument('--data_directory', dest="data_directory", default="flowers", type=str)

    args = parser.parse_args()
    #print(args)

    # Setup devivce and load checkpoint
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        checkpoint = torch.load(args.checkpoint)
    else:
        device = torch.device("cpu")
        # https://gist.github.com/jzbontar/b50f8c9dd22e49ff68c7c91dad63166a
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    # Load te model
    model = train.build_network(checkpoint['hidden_layers'], checkpoint['drop_p'], checkpoint['model_network'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Predict categories
    if args.top_k is not None:
        topk = args.top_k
    else:
        topk = 5

    probs, labels = predict(args.image_path, model, args.gpu, topk)

    #  Print most likely image class and it's associated probability
    print("Most likely image class: {}".format(labels[0]))
    print("Probability of this image class: {}%".format( round(100 * probs[0], 2)))
    # Print top K classes along with associated probabilities
    if args.top_k is not None:
        index = 0
        print("-"*20)
        print("Top {} categories and their probabilities:".format(args.top_k))
        for i in np.nditer(probs):
            print("{} : {}%".format( labels[index], round(100 * i, 2) ))
            index +=1

    # load a JSON file that maps the class values to other category names
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        print("-"*20)
        print("Top {} flowers categories and their probabilities:".format(args.top_k))
        index = 0
        for i in np.nditer(probs):
            print("{} : {}%".format( cat_to_name[labels[index]].title(), round(100 * i, 2) ))
            index +=1

if __name__== "__main__":
  main()

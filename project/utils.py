import numpy as np
import os
from PIL import Image
from torchvision import models, transforms
import torch

def get_features(data_dir, save_file='./features'):

    num_images = len(os.listdir(data_dir))

    trans = transforms.Compose([    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize( [0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225] ) ])

    model1 = models.resnet50(pretrained=True)
    new_model = torch.nn.Sequential(*list(model1.children())[:-2])

    new_model.eval()

    array = torch.randn(3,224,224)
    new_preds = new_model(array.unsqueeze(0))
    nparr = new_preds.detach().numpy()
    features = []
    names = []
    files = sorted(os.listdir(data_dir))

    for i,pic in enumerate(files):
        try:
            img = Image.open(data_dir+pic)
            img = trans(img)
            new_preds = new_model(img.unsqueeze(0))
            nparr = new_preds.detach().numpy()
            num_features = nparr.shape[1]
            vect = list()
            for max in nparr[0]:
                vect.append(max.max())
            vect = np.array(vect)
            print(pic)
            names.append(pic)
            features.append(vect)

        except:
            print("Error opening: ",data_dir,pic)

    np.save(save_file,features)

    return features


# Manually compute mean
def mean(data):
    sum = 0.0
    for val in data:
        sum += val

    return sum/len(data)


# Manually compute variance
def var(data):
    m = mean(data)

    sum = 0.0
    for val in data:
        sum += np.power(val - m, 2)

    return sum/len(data)


# Manually compute std
def std(data):
    va = var(data)
    return np.power(va, .5)


# Pretty print for matrices
def print_c(str,mats):
    print(str)
    for c in mats:
        print(c)
    print()

    
def feature_selection(data, features, proj_dim):
    print('Using Feature Selection')
    # Smush features together
    features_ravel = np.empty((1,2048))
    for feat in data:
        features_ravel = np.append(features_ravel, feat, axis=0)
    features_ravel = features_ravel[1:]

    new_data = np.empty((features_ravel.shape[0], 1))
    for feat,i in zip(features,range(proj_dim)):
        new_data = np.append(new_data, features_ravel[:,feat].reshape((features_ravel.shape[0], 1)), axis=1)
    
    new_data = new_data[:,1:].T

    # Center Data
    for i in range(new_data.shape[0]):
        m = mean(new_data[i])
        new_data[i] -= m
    
    return new_data.T


def no_dimensional_reduction(data):
    # Unravel
    features_ravel = np.empty((1,2048))
    for feat in data:
        features_ravel = np.append(features_ravel, feat, axis=0)
    features_ravel = features_ravel[1:]

    features_ravel = features_ravel[:,1:].T

    # Center Data
    for i in range(features_ravel.shape[0]):
        m = mean(features_ravel[i])
        features_ravel[i] -= m

    return features_ravel.T
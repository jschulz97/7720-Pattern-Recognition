import numpy as np
import os
from PIL import Image
from torchvision import models, transforms
import torch
import yaml
import requests

config_file_name = 'config.yaml'


class resnet():
    def __init__(self, ):
        self.trans = transforms.Compose([    transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize( [0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225] ) ])

        model1 = models.resnet50(pretrained=True)
        #new_model = torch.nn.Sequential(*list(model1.children())[:-2])
        self.new_model = model1

        self.new_model.eval()

        # Let's get our class labels.
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
        self.labels = {int(key): value for key, value in response.json().items()}


    def __call__(self, data_dir):
        num_images = len(os.listdir(data_dir))
        features = []
        names = []
        files = sorted(os.listdir(data_dir))

        for i,pic in enumerate(files):
            try:
                img = Image.open(data_dir+pic)
                img = self.trans(img)
                new_preds = self.new_model(img.unsqueeze(0))
                nparr = new_preds.detach().numpy()
                print(pic)
                names.append(pic)
                # features.append(self.labels[np.argmax(nparr)])
                features.append(np.argmax(nparr))

            except:
                print("Error opening: ",data_dir,pic)

        return features


if(__name__ == '__main__'):
    #####################################################
    # Load Config
    #####################################################
    print('\nAttempt: Loading', config_file_name+'...')
    try:
        with open(config_file_name) as config_file:
            params = yaml.load(config_file, Loader=yaml.FullLoader)
    except:
        print('\nconfig.yaml not in working directory! Exiting.\n')
        exit()

    print('Configuration loaded!')

    y_test = np.array([292] * 50)
    y_test = np.append(y_test, [293] * 50)

    model = resnet()
    preds = []
    for dir in params['test_data_dir']:
        preds.append(model(dir))
    preds = np.array(preds).ravel()

    score = 0.0
    for p, y in zip(preds, y_test):
        if(p == y):
            score += 1.0

    print('\nPrediction Score:', str(int(score))+'/'+str(len(y_test)), '=', score/len(y_test))
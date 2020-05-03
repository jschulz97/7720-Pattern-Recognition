import numpy as np
import plotly.graph_objects as go
import yaml

import utils
import lda
# import qda
import pca
import svm

config_file_name = 'config.yaml'


#####################################################
# Load Config
#####################################################
try:
    print('\nAttempt: Loading', config_file_name+'...')
    with open(config_file_name) as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)
    print('Configuration loaded!')
except:
    print('\nconfig.yaml not in working directory! Exiting.\n')
    exit()


#####################################################
# Get Features
#####################################################
print('\nAttempt: Loading features...')
features = []

if(params['get_pytorch_features']):
    for src, dest in zip(params['data_dir'], params['features_dir']):
        features.append(utils.get_features(src, dest))

else:    
    for src, dest in zip(params['data_dir'], params['features_dir']):
        features.append(np.load(dest+'.npy'))

print('Features loaded!')


#####################################################
# Dimensional Reduction
#####################################################
if(params['dim_red_choice'] == 'pca'):
    dim_red = pca.PCA()
elif(params['dim_red_choice'] == 'lda'):
    dim_red = lda.LDA()
elif(params['dim_red_choice'] == 'qda'):
    dim_red = qda.QDA()
else:
    print('\nError: Select a valid dimensional reduction!')
    exit()

data_reduced = dim_red.cheat(features, params['projection_dim'])


#####################################################
# Display Reductions
#####################################################








#####################################################
# Classify
#####################################################
# Create labels
y = np.array([0] * 100)
y = np.append(y, [1] * 100)
y = np.append(y, [2] * 100)

classifier = svm.SVM()
W = classifier.learn(data_reduced, y) 







#####################################################
# Results
#####################################################




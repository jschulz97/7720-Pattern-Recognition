import numpy as np
import plotly.graph_objects as go
import yaml

import utils
import lda
import qda
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
# PCA
if(params['dim_red_choice'] == 'pca'):
    dim_red = pca.PCA_1()

# LDA

# QDA

features_ravel = np.empty((1,2048))
for feat in features:
    features_ravel = np.append(features_ravel, feat, axis=0)
features_ravel = features_ravel[1:]

dim_red(features_ravel, 2)




#####################################################
# Display Reductions
#####################################################








#####################################################
# Classify
#####################################################








#####################################################
# Results
#####################################################




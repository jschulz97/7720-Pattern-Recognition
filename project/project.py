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
print('\nAttempt: Loading', config_file_name+'...')
try:
    with open(config_file_name) as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)
except:
    print('\nconfig.yaml not in working directory! Exiting.\n')
    exit()

print('Configuration loaded!')


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
print('\nAttempt: Dimensional Reduction...')
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

print('Finished!')


#####################################################
# Display Reductions
#####################################################
if(params['display_projection']):
    if(params['projection_dim'] == 2):
        fig = go.Figure()
        for i in range(0,200,100):
            fig.add_trace(
                go.Scatter(x=data_reduced.T[0][i:i+100], y=data_reduced.T[1][i:i+100], mode='markers')
            )
        
        fig.show()

    elif(params['projection_dim'] == 3):
        fig = go.Figure()
        for i in range(0,200,100):
            fig.add_trace(
                go.Scatter3d(x=data_reduced.T[0][i:i+100], y=data_reduced.T[1][i:i+100], z=data_reduced.T[2][i:i+100], mode='markers', marker=dict(size=3))
            )

        fig.show()


#####################################################
# Classify
#####################################################
print('\nAttempt: Classification...')
# Create labels
y = np.array([-1] * 100)
y = np.append(y, [1] * 101)

classifier = svm.SVM()
W = classifier.learn(data_reduced, y) 
# W = np.array([ 2.10110659,-1.08379617])

svm_trans = data_reduced * W



print('Finished!')


#####################################################
# Results
#####################################################
if(params['display_projection']):
    # Calculate line coords
    m = W[0] / W[1]
    if(m == abs(m)):
        x0 = min(svm_trans[:,0])
        x1 = max(svm_trans[:,0])
        while(x0 * m < min(svm_trans[:,1])):
            x0 += 1.0
        while(x1 * m > max(svm_trans[:,1])):
            x1 -= 1.0
        x0 -= 1.0
        x1 += 1.0
    else:
        x0 = min(svm_trans[:,0])
        x1 = max(svm_trans[:,0])
        while(x1 * m < min(svm_trans[:,1])):
            x1 -= 1.0
        while(x0 * m > max(svm_trans[:,1])):
            x0 += 1.0
        x0 -= 1.0
        x1 += 1.0

    y0 = x0 * m
    y1 = x1 * m

    if(params['projection_dim'] == 2):
        

        fig = go.Figure()
        for i in range(0,300,100):
            fig.add_trace(
                go.Scatter(x=svm_trans.T[0][i:i+100], y=svm_trans.T[1][i:i+100], mode='markers')
            )
        fig.add_shape(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(
                    color="RoyalBlue",
                    width=3
                )
            )
        )
        
        fig.show()

    elif(params['projection_dim'] == 3):
        mesh_x, mesh_y = np.meshgrid(np.arange(x0, x1, 1.0), np.arange(y0, y1, 1.0))
        mesh_z = np.array(mesh_x.shape)
        print(mesh_x.shape, mesh_y.shape)
        for x in mesh_x[0]:
            for y in mesh_y[1]:
                _=0

        fig = go.Figure()
        for i in range(0,300,100):
            fig.add_trace(
                go.Scatter3d(x=svm_trans.T[0][i:i+100], y=svm_trans.T[1][i:i+100], z=svm_trans.T[2][i:i+100], mode='markers', marker=dict(size=3))
            )
        
        fig.add_shape(
        # filled Rectangle
            type="rect",
            x0=-10,
            y0=-10,
            z0=-10,
            x1=10,
            y1=10,
            z1=10,
            line=dict(
                color="RoyalBlue",
                width=2,
            ),
            fillcolor="LightSkyBlue",
        )

        fig.show()



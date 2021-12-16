import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yaml

import utils
import lda
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
test_features = []

if(params['get_pytorch_features_train']):
    for src, dest in zip(params['data_dir'], params['features_dir']):
        features.append(utils.get_features(src, dest))

else:    
    for src, dest in zip(params['data_dir'], params['features_dir']):
        features.append(np.load(dest+'.npy'))

# Create labels
if('cheetah' in params['features_dir'][0] and 'monkey' in params['features_dir'][1]):
    print('FLIPPED TRAIN LABELS')
    y_train = np.array([1] * 100)
    y_train = np.append(y_train, [-1] * 100)
else:
    y_train = np.array([-1] * 100)
    y_train = np.append(y_train, [1] * 100)

# Testing data
if(params['resub']):
    test_features = features
    test_labels   = y_train
else:
    if(params['get_pytorch_features_test']):
        for src, dest in zip(params['test_data_dir'], params['test_features_dir']):
            test_features.append(utils.get_features(src, dest))

    else:    
        for src, dest in zip(params['test_data_dir'], params['test_features_dir']):
            test_features.append(np.load(dest+'.npy'))

    # Create labels
    if('cheetah' in params['test_features_dir'][0] and 'monkey' in params['test_features_dir'][1]):
        print('FLIPPED TEST LABELS')
        y_test = np.array([1] * 50)
        y_test = np.append(y_test, [-1] * 50)
    else:
        y_test = np.array([-1] * 50)
        y_test = np.append(y_test, [1] * 50)

print('Features loaded!')


#####################################################
# Dimensional Reduction
#####################################################
print('\nAttempt: Dimensional Reduction...')
reduce = True
if(params['dim_red_choice'] == 'pca'):
    dim_red = pca.PCA()
elif(params['dim_red_choice'] == 'lda'):
    dim_red = lda.LDA()
elif(params['dim_red_choice'] == 'qda'):
    dim_red = qda.QDA()
elif(params['dim_red_choice'] == 'selection'):
    data_reduced = utils.feature_selection(features, params['feature_selections'], params['projection_dim'])
    reduce = False
elif(params['dim_red_choice'] == 'none'):
    data_reduced = utils.no_dimensional_reduction(features)
    reduce = False
else:
    print('\nError: Select a valid dimensional reduction!')
    exit()

if(reduce):
    data_reduced = dim_red.cheat(features, params['projection_dim'], labels=y_train)

print('Dimensional Reduction finished!')


#####################################################
# Display Reductions
#####################################################
if(params['display_projection'] and params['dim_red_choice'] != 'none'):
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

# Init classifier
classifier = svm.SVM()

# Fit to training data
W = classifier.learn(data_reduced, y_train)

if(params['resub']):
    # Classify on resub
    classifier.predict(data_reduced, y_train)
else:
    # Classify on test data

    # Smush features together
    features_ravel = np.empty((1,2048))
    for feat in test_features:
        features_ravel = np.append(features_ravel, feat, axis=0)
    features_ravel = features_ravel[1:]

    preds = dim_red.predict(features_ravel)

    score = 0.0
    for p, l in zip(preds, y_test):
        if(p == l):
            score += 1.0
    
    print('\nPrediction Score:', str(int(score))+'/'+str(len(y_test)), '=', score/len(y_test))


print('Classification finished!')


#####################################################
# Results
#####################################################
if(params['display_projection'] and params['dim_red_choice'] != 'none'):
    svm_trans = data_reduced * W

    if(params['projection_dim'] == 1):
        hist_data = []
        for i in range(0,200,100):
            hist_data.append(svm_trans[i:i+100, 0])
        group_labels = ['-1', '1']

        # Create distplot
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
        fig.show()

    elif(params['projection_dim'] == 2):
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
                    color="Green",
                    width=3
                )
            )
        )
        
        fig.show()

    elif(params['projection_dim'] == 3):
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
        
        fig.show()



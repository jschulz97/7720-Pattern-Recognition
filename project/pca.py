import utils
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


class PCA_1:
    def __init__(self,):
        _=0

    def __call__(self, data, proj_dim):
        if(data.shape[1] <= proj_dim):
            print('Invalid dimensions for projection.')
            exit()

        data_T = data.T

        # For every dimension
        for i in range(data_T.shape[0]):
            # Center
            m = utils.mean(data_T[i])
            data_T[i] -= m

            # # Std = 1
            # s = utils.std(data_T[i])
            # data_T[i] /= s

        # Create Scatter (Covariances)
        S   = np.dot(data_T, data_T.T)
        mx  = np.max(S)
        S   /= mx

        # print(S.shape)
        # utils.print_c('Covs from hand',S)

        # Calculate Eigenvalues and Eigenvectors
        eva, eve = np.linalg.eig(S)

        # utils.print_c('evects',eve)

        # Sort by largest eigenvalues
        eva_s = np.sort(eva)[::-1]
        eva_i = np.argsort(eva)[::-1]
        
        eve_s = np.zeros(eve.shape)
        for i, e_i in enumerate(eva_i):
            eve_s[i] = eve[e_i]

        # utils.print_c('evects sorted',eve_s)

        # Pick features
        eve_s = eve_s[:proj_dim]

        # Transform
        transform = np.dot(data_T.T, eve_s.T)

        print(transform.shape)
        
        # fig = go.Figure(data=[
        #     go.Scatter3d(x=data.T[0], y=data.T[1], z=data.T[2], mode='markers', marker=dict(size=3)),
        #     go.Scatter3d(x=transform.T[0], y=transform.T[1], z=transform.T[2], mode='markers', marker=dict(size=3)),
        #     ])
        # fig.show()

        # fig = make_subplots(rows=1,cols=2)

        # for i in range(0,300,100):
        #     fig.add_trace(
        #         go.Scatter(x=data.T[0][i:i+100], y=data.T[1][i:i+100], mode='markers'),
        #         row=1, col=1
        #     )

        # for i in range(0,300,100):
        #     fig.add_trace(
        #         go.Scatter(x=transform.T[0][i:i+100], y=transform.T[1][i:i+100], mode='markers'),
        #         row=1, col=2
        #     )


        fig = go.Figure()
        for i in range(0,300,100):
            fig.add_trace(
                go.Scatter(x=transform.T[0][i:i+100], y=transform.T[1][i:i+100], mode='markers'))
        
        fig.show()

        pca1 = PCA(n_components=2)
        X_new = pca1.fit_transform(data)

        fig = go.Figure()
        for i in range(0,300,100):
            fig.add_trace(
                go.Scatter(x=X_new.T[0][i:i+100], y=X_new.T[1][i:i+100], mode='markers'))
        
        fig.show()

        # asdf


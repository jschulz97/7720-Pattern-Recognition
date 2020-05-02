import utils
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda_cheat


class QDA:
    def __init__(self,):
        _=0
    
    def __call__(self, data, proj_dim):
        print('\nUsing QDA Manual...\n')
        _=0


    def cheat(self, data, proj_dim):
        # Cheat
        print('\nUsing QDA Cheat...\n')

        qda = qda_cheat(n_components=proj_dim)
        X_new = qda.fit_transform(data)

        if(proj_dim == 2):
            fig = go.Figure()
            for i in range(0,300,100):
                fig.add_trace(
                    go.Scatter(x=X_new.T[0][i:i+100], y=X_new.T[1][i:i+100], mode='markers')
                )
            
            fig.show()
        
        elif(proj_dim == 3):
            # 3d Scatter
            fig = go.Figure()
            for i in range(0,300,100):
                fig.add_trace(
                    go.Scatter3d(x=X_new.T[0][i:i+100], y=X_new.T[1][i:i+100], z=X_new.T[2][i:i+100], mode='markers'), marker=dict(size=3)
                )

            fig.show()
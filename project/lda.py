import utils
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda_cheat


class LDA:
    def __init__(self,):
        _=0

    def __call__(self, data, proj_dim):
        print('Using LDA Manual')

        # means = []
        # for d in data:
        #     means.append(utils.mean(d))

        # me
        # mean_all = df2.mean(axis = 0)

        # mean_1 = m_1.reshape(1,19)
        # mean_1 = np.repeat(mean_1,20,axis = 0)

        # mean_2 = m_2.reshape(1,19)
        # mean_2 = np.repeat(mean_2,20,axis = 0)

        # within_class_scatter = np.zeros((19,19))
        # wcs_1 = np.zeros((19,19))
        # wcs_1 = np.matmul((np.transpose(df3_1 - mean_1 )), (df3_1 - mean_1))

        # wcs_2 = np.zeros((19,19))
        # wcs_2 = np.matmul((np.transpose(df3_2 - mean_2 )), (df3_2 - mean_2))

        # within_class_scatter = np.add(wcs_1,wcs_2)

        # bcs_1 = np.multiply(len(df3_1),np.outer((m_1 - mean_all),(m_1 - mean_all)))
        # bcs_2 = np.multiply(len(df3_2),np.outer((m_2 - mean_all),(m_2 - mean_all)))

        # between_class_scatter = np.add(bcs_1,bcs_2)

        # e_val, e_vector = np.linalg.eig(np.dot(lg.inv(within_class_scatter),between_class_scatter))
        # for e in range (len(e_val)):
        #     e_scatter = e_vector[:,e].reshape(19,1)

        #     print(e_val[e].real)

        # print(between_class_scatter)


        # eig_pairs = [(np.abs(e_val[i]).real, e_vector[:,i].real) for i in range(len(e_val))]


        # eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)



        # print('Eigenvalues in decreasing order:\n')
        # for i in eig_pairs:
        #     print(i[0])

        # W= eig_pairs[0][1].reshape(19,1)


    def cheat(self, data, proj_dim, labels=None):
        # Cheat
        print('Using LDA Cheat')

        # Smush features together
        features_ravel = np.empty((1,2048))
        for feat in data:
            features_ravel = np.append(features_ravel, feat, axis=0)
        features_ravel = features_ravel[1:]

        self.lda = lda_cheat(n_components=proj_dim, solver='svd')
        X_new = self.lda.fit_transform(features_ravel, labels)
        
        return X_new

    
    def predict(self, data):
        return self.lda.predict(data)
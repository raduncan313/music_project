import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import pickle
import os
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.embeddings = {}
        self.clusterings = {}
        
    def __str__(self):
        model_str = f'Models:\n{''.join([n + ': ' + str(self.models[n]) + '\n' for n in self.models.keys()])}'
        cluster_str = f'Clusters:\n{''.join([n + ': ' + str(self.clusterings[n]) + '\n' for n in self.clusterings.keys()])}'
        embed_str = f'Embeddings:\n{''.join([n + ': ' + str(self.embeddings[n][0]) + '\n' for n in self.embeddings.keys()])}'
        return model_str + '\n\n' + cluster_str + '\n\n' + embed_str
    
    def train_test_split(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    def train_model_gscv(self, model, name, params, random_state):
        clf = model(random_state=random_state)
        gscv = GridSearchCV(clf, params, verbose=3, n_jobs=-1)
        gscv.fit(self.X_train, self.y_train)
        exec(f'self.models[\'{name}\'] = gscv')
        print(f'Scores for \'{name}\':')
        print(f'Training score: {self.training_score(name)}')
        print(f'Test score: {self.test_score(name)}')
        print(' ')
        print('Best params:')
        print(self.models[name].best_params_)
        print(' ')
        
    def train_tree_ccp(self, name, val_size, random_state):
        Xt2, Xv, yt2, yv = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=random_state)
        clf = DecisionTreeClassifier(random_state=random_state)
        path = clf.cost_complexity_pruning_path(Xt2, yt2)
        alphas = path.ccp_alphas
        
        clfs = []
        for al in alphas:
            clf = DecisionTreeClassifier(ccp_alpha=al, random_state=random_state)
            clf.fit(Xt2, yt2)
            clfs.append(clf)
        
        t2_scores = [clf.score(Xt2, yt2) for clf in clfs]
        v_scores = [clf.score(Xv, yv) for clf in clfs]
        
        v_tup = zip(clfs, v_scores)
        clf_opt = max(v_tup, key=lambda x: x[1])[0]
        clf_opt.fit(self.X_train, self.y_train)
        exec(f'self.models[\'{name}\'] = clf_opt')
        print(f'Scores for \'{name}\':')
        print(f'Training score: {self.training_score(name)}')
        print(f'Test score: {self.test_score(name)}')
        print(' ')
        print(f'ccp_alpha: {self.models[name].ccp_alpha}')
        print(f'max_depth: {self.models[name].max_depth}')
        print(f'min_samples_split: {self.models[name].min_samples_split}')
        print(f'min_samples_leaf: {self.models[name].min_samples_leaf}')
        print(' ')
    
    class Projection:
        def __init__(self, X, y, infer):
            if infer:
                X = X.copy()
                s = 1 - X.sum(axis=1)
                X.insert(0, 'infer', s)
                
            df = pd.concat([y, X], axis=1)
            self.M = df.groupby(y.name).mean()
            self.infer = infer
        
        def score(self, X, y):
            if self.infer:
                X = X.copy()
                s = 1 - X.sum(axis=1)
                X.insert(0, 'infer', s)
                
            y_pred = (X @ self.M.T).idxmax(axis=1)
            return (y == y_pred).mean()
        
    def train_projection(self, name, infer=True):
        proj = self.Projection(self.X_train, self.y_train, infer)
        exec(f'self.models[\'{name}\'] = proj')
        print(f'Scores for \'{name}\':')
        print(f'Training score: {self.training_score(name)}')
        print(f'Test score: {self.test_score(name)}')
        print(' ')
        
        
    def test_score(self, name):
        model = self.models[name]
        if isinstance(model, GridSearchCV):
            return model.best_estimator_.score(self.X_test, self.y_test)
        else:
            return model.score(self.X_test, self.y_test)
    
    def training_score(self, name):
        model = self.models[name]
        if isinstance(model, GridSearchCV):
            return model.best_estimator_.score(self.X_train, self.y_train)
        else:
            return model.score(self.X_train, self.y_train)
    
    def delete_model(self, name):
        del self.models[name]
        
    def fit_kmeans(self, name, random_state):
        inits = ['k-means++', 'random']
        best_score = -np.inf
        best_kmc = None
        n_clusters = self.y.nunique()
        for init in inits:
            n_init = 100 if init == 'random' else 1
            kmc = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
            y_pred = kmc.fit_predict(self.X)
            score = adjusted_mutual_info_score(self.y, y_pred)
            if score > best_score:
                best_kmc = kmc
                best_score = score
        exec(f'self.clusterings[\'{name}\'] = best_kmc')
        self.report_cluster(name)
    
    def fit_gmm(self, name, random_state):
        init_params_list = ['kmeans', 'k-means++', 'random_from_data']
        best_score = -np.inf
        best_gmm = None
        n_components=self.y.nunique()
        for init_params in init_params_list:
            n_init = 100 if init_params == 'random_from_data' else 1
            gmm = GaussianMixture(n_components=n_components, init_params=init_params, n_init=n_init, random_state=random_state)
            y_pred = gmm.fit_predict(self.X)
            score = adjusted_mutual_info_score(self.y, y_pred)
            if score > best_score:
                best_gmm = gmm
                best_score = score
        best_gmm.labels_ = best_gmm.predict(self.X)
        exec(f'self.clusterings[\'{name}\'] = best_gmm')
        self.report_cluster(name)
    
    def fit_dbscan(self, name, eps_vals, min_samples_vals, random_state):
        best_score = -np.inf
        best_db = None
        for eps in eps_vals:
            for min_samples in min_samples_vals:
                db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                y_pred = db.fit_predict(self.X)
                score = adjusted_mutual_info_score(self.y, y_pred)
                if score > best_score:
                    best_score = score
                    best_db = db
        exec(f'self.clusterings[\'{name}\'] = best_db')
        self.report_cluster(name)
        print(f'Best eps: {self.clusterings[name].eps}')
        print(f'Best min_samples: {self.clusterings[name].min_samples}')
        
    def fit_tsne(self, name, random_state, init='pca'):
        tsne = TSNE(init=init, random_state=random_state)
        Xe = tsne.fit_transform(self.X)
        exec(f'self.embeddings[\'{name}\'] = [tsne, Xe]')
    
    def get_embedding(self, name):
        return self.embeddings[name][1]
    
    def plot_embedding(self, name):
        fig, ax = plt.subplots()
        Xe = self.embeddings[name][1]
        ax.scatter(Xe[:,0], Xe[:,1], c=self.y, cmap='jet')
        return fig, ax
    
    def report_cluster(self, name):
        print(f'Best clustering: {self.clusterings[name]}')
        print(f'Info score for cluster: {self.info_score(name)}')
        
    def info_score(self, name):
        return adjusted_mutual_info_score(self.y, self.clusterings[name].labels_)
    
def plot_tsne(ax, Xe, y, cmap=plt.cm.viridis, legend_dict=None, markersize=3):
    cols = cmap(np.linspace(0,1,len(np.unique(y))))
    for j,v in enumerate(np.unique(y)):
        filt = (y == v)
        X = Xe[filt, :]
        if legend_dict:
            w = legend_dict[v]
        else:
            w = v
        ax.plot(X[:,0], X[:,1], 'o', color=cols[j], label=w, markersize=markersize)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    return ax
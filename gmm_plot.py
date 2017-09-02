import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.mixture import GMM

colors = ['darkorange', 'navy', 'turquoise', 'cornflowerblue']
color_mapping = dict(zip(range(len(colors)), colors))


def make_ellipses(gmm, ax):
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covars_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covars_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covars_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covars_[n]

        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color_mapping[n])

        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def plot_gmm(n_components, dataframe):
    plt.figure(figsize=(12, 12))

    for index, covariance_type in enumerate(['spherical', 'tied', 'diag', 'full']):
        gmm = GMM(n_components=n_components, covariance_type=covariance_type).fit(dataframe)
        pred = gmm.predict(dataframe)
        ax = plt.subplot(2, 2, index + 1)
        ax.axis('equal')
        make_ellipses(gmm, ax)
        colors = np.vectorize(color_mapping.get)(pred)
        plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], color=colors)
        plt.title(covariance_type)


if __name__ == '__main__':
    plot_gmm(4, pd.DataFrame())  # TODO: Put REAL dataframe in here

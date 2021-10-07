import numpy as np
from sklearn.kernel_approximation import RBFSampler
from scipy.linalg import lstsq


class RandomFourierFeatures(object):
    def __init__(self, n_features: int = 20, gamma: float = 1.0, random_state: int = 0):
        self.gamma = gamma
        self.rbf_feature = RBFSampler(n_components=n_features, gamma=gamma, random_state=random_state)
        self.n_features = n_features
        self.random_state = random_state
        self.estim_param = None

    def fit(self, X, y):
        X = np.atleast_2d(X)
        Xf = self.rbf_feature.fit_transform(X)
        self.estim_param, _resid, _rank, _s = lstsq(Xf, y)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        Xf = self.rbf_feature.transform(X)
        return Xf @ self.estim_param

    @property
    def param_norm(self):
        return np.linalg.norm(self.estim_param)



if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    n_experiments = 100

    # load and split dataset
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

    n_train = y_train.shape[0]

    n_features = np.zeros(n_experiments,dtype=int)
    mse_train = np.zeros(n_experiments)
    mse_test = np.zeros(n_experiments)
    param_norm = np.zeros(n_experiments)

    # Repeat experiments for different number of parameters
    for i, fraction in enumerate(np.logspace(-1, 1, n_experiments)):
        # initialize and train model
        n_features[i] = int(fraction * n_train)
        mdl = RandomFourierFeatures(n_features=n_features[i])
        mdl.fit(X_train, y_train)
        y_train_pred = mdl.predict(X_train)

        # Evaluate on test
        y_test_pred = mdl.predict(X_test)

        # compute mse
        mse_train[i] = np.mean((y_train_pred - y_train)**2)
        mse_test[i] = np.mean((y_test_pred - y_test) ** 2)
        param_norm[i] = mdl.param_norm

    fig, ax = plt.subplots(2)
    ax[0].plot(n_features / n_train, mse_train, label='train')
    ax[0].plot(n_features / n_train, mse_test, label='test')
    ax[0].axvline(1, ls='--')
    ax[0].set_ylabel('Mean square error')
    ax[0].set_xscale('log')
    ax[0].set_ylim((0, 5000))
    ax[0].legend()
    ax[1].plot(n_features / n_train, param_norm)
    ax[1].set_ylabel('Parameter norm')
    ax[1].set_xlabel('# features / # datapoints')
    ax[1].set_xscale('log')
    ax[1].axvline(1, ls='--')
    ax[1].set_ylim((0, 5000))
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch.random
import torch.nn as nn
import torch.optim as optim
import tqdm
from scipy.stats import multivariate_normal
import scipy.linalg as linalg

EPS = 1e-9


class FeedforwardNetwork(nn.Module):
    """Neural network implemented using NumPy."""

    def __init__(self, input_size, width=2000, beta=0.1, depth=2, only_optimize_last=False):
        super(FeedforwardNetwork, self).__init__()
        self.width = width
        self.beta = beta
        self.depth = depth
        self.ws = [input_size] + [self.width for _l in range(self.depth)] + [1]

        self.weights = []
        self.biases = []
        for i in range(depth + 1):
            wi = nn.Parameter(torch.randn(self.ws[i], self.ws[i + 1]))
            bi = nn.Parameter(torch.randn(self.ws[i + 1]))
            if (not only_optimize_last) or (i == depth):
                self.register_parameter('weight_{}'.format(i), wi)
                self.register_parameter('bias_{}'.format(i), bi)
            self.weights += [wi]
            self.biases += [bi]

    def forward(self, X):
        """Return output and gradient"""
        Xf = self.get_features(X)
        pred = 1/np.sqrt(self.ws[-2]) * Xf @ self.weights[-1] + self.beta * self.biases[-1]
        return pred.flatten()

    def get_features(self, X):
        activ = X
        for i in range(self.depth):
            preactiv = 1/np.sqrt(self.ws[i]) * activ @ self.weights[i] + self.beta * self.biases[i]
            activ = torch.relu(preactiv)
        return activ

def train_and_predict(X_train, Y_train, X_test, net, epochs=1000, lr=1.0, sgd=False):
    # Convert to pytorch
    X_train_pth = torch.Tensor(X_train)
    Y_train_pth = torch.Tensor(Y_train)
    X_test_pth = torch.Tensor(X_test)
    # Train
    msg = 'Epoch = {} - loss = {:0.2f}'
    if sgd:
        pbar = tqdm.tqdm(initial=0, total=epochs, desc=msg.format(0, 0))
        optimizer = optim.SGD(net.parameters(), lr=lr)
        for i in range(epochs):
            optimizer.zero_grad()
            Y_pred = net(X_train_pth)
            error = (Y_train_pth - Y_pred)
            loss = 1/2 * (error * error).mean()
            loss.backward()
            optimizer.step()
            # Update pbar
            pbar.desc = msg.format(i, loss)
            pbar.update(1)
        pbar.close()
        # Predict on test
        with torch.no_grad():
            Y_pred_test = net(X_test_pth).detach().numpy()
    else:
        with torch.no_grad():
            # Train
            Xf_train = net.get_features(X_train_pth).detach().numpy()
            Xf_train = np.hstack([Xf_train, np.ones([Xf_train.shape[0], 1])])
            estim_param, _resid, _rank, _s = linalg.lstsq(Xf_train, Y_train)
            # Test
            Xf_test = net.get_features(X_test_pth).detach().numpy()
            Xf_test = np.hstack([Xf_test, np.ones([Xf_test.shape[0], 1])])
            Y_pred_test = Xf_test @ estim_param
    return Y_pred_test


def linkernel(x, beta=1):
    n_pts, dim = x.shape
    return 1 / dim * x @ x.T + beta**2 + EPS * np.eye(n_pts)


def approximate_kernel(C, fn, n_samples=1000):
    y = multivariate_normal.rvs(size=n_samples, cov=C)
    return 1 / n_samples * fn(y.T)  @ fn(y)


def nngp_cov(x, n_layers, beta=1):
    sigma = linkernel(x, beta=beta)
    relu = lambda xx: np.maximum(xx, 0)
    for _i in range(n_layers):
        sigma = approximate_kernel(sigma, relu) + beta ** 2
    return sigma


def ntk_cov(x, n_layers, beta=1):
    sigma = linkernel(x, beta)
    theta = sigma

    relu = lambda xx: np.maximum(xx, 0)
    relu_deriv = lambda xx: np.array(xx > 0, dtype=float)
    for _i in range(n_layers):
        sigma = approximate_kernel(sigma, relu) + beta ** 2
        sigma_dot = approximate_kernel(sigma, relu_deriv)
        theta = theta * sigma_dot + sigma
    return sigma, theta


def conditioning(c, y_train):
    n_train = len(y_train)
    if n_train == 0:
        return np.zeros(c.shape[0]), c
    cov_train = c[:n_train, :n_train]
    cov_test_train = c[n_train:, :n_train]
    cov_test = c[n_train:, n_train:]

    inv_cov = np.linalg.inv(cov_train)
    m = cov_test_train @ inv_cov @ y_train
    cc = cov_test - cov_test_train @ inv_cov @ cov_test_train.T
    return m, cc


def ntk_limit(c, ntkc, y_train):
    n_train = len(y_train)
    if n_train == 0:
        return np.zeros(c.shape[0]), c
    # Get submatrices
    cov_train = c[:n_train, :n_train]
    cov_test_train = c[n_train:, :n_train]
    cov_test = c[n_train:, n_train:]
    # Get ntk submatrices
    ntk_cov_train = ntkc[:n_train, :n_train]
    ntk_cov_test_train = ntkc[n_train:, :n_train]
    inv_ntkc = np.linalg.inv(ntk_cov_train)

    m = ntk_cov_test_train @ inv_ntkc @ y_train
    cc = cov_test + ntk_cov_test_train @ inv_ntkc @ cov_train @ inv_ntkc @ ntk_cov_test_train.T + \
          - ntk_cov_test_train @ inv_ntkc @ cov_test_train.T - cov_test_train @ inv_ntkc @ ntk_cov_test_train.T
    return m, cc


if __name__ == "__main__":
    N_test = 50
    n_layers = 4
    beta = 0.1
    epochs = 1000
    width = 1000
    lr = 1
    tp = 'nngp'  # CHANGE HERE: 'nngp' gives part 1; 'ntk' gives part 2.
    n_runs = 5

    # Train data
    gamma_train = np.array([-2, -1.2, -0.4, 0.5, 1.8])
    X_train = np.stack([np.cos(gamma_train), np.sin(gamma_train)]).T
    Y_train = X_train.prod(axis=1)

    # Test data
    gamma_test = np.linspace(-np.pi, np.pi, N_test)
    X_test = np.stack([np.cos(gamma_test), np.sin(gamma_test)]).T
    Y_test = X_test.prod(axis=1)

    # Define neural network
    Y_test_pred = []
    for i in range(n_runs):
        tqdm.tqdm.write('run={}'.format(i))
        torch.manual_seed(i)
        net = FeedforwardNetwork(2, beta=beta, depth=n_layers, width=width,
                                 only_optimize_last=(tp=='nngp'))
        Y_test_pred += [train_and_predict(X_train, Y_train, X_test, net, lr=lr, epochs=epochs,
                                          sgd=(tp!='nngp'))]

    # Plot on test data
    for i, y in enumerate(Y_test_pred):
        plt.plot(gamma_test, y, color='black')
    plt.plot(gamma_test, Y_test)
    plt.plot(gamma_train, Y_train, '*', color='black', ms=10)

    # Compute covariance matrix
    if tp == 'nngp':
        c = nngp_cov(np.vstack([X_train, X_test]), n_layers=n_layers, beta=beta)
        m, cc = conditioning(c, Y_train)
    else:
        c, ntkc = ntk_cov(np.vstack([X_train, X_test]), n_layers=n_layers, beta=beta)
        m, cc = ntk_limit(c, ntkc, Y_train)
    y_gp = multivariate_normal.rvs(size=5, mean=m.flatten(), cov=cc)
    plt.plot(gamma_test, y_gp.T, color=str(0.4), alpha=0.8)
    for nn in [3, 2, 1]:
        plt.fill_between(gamma_test, m - nn * np.sqrt(np.diag(cc)),
                         m + nn * np.sqrt(np.diag(cc)), color=str(0.4 + 0.15 * nn), alpha=0.5)
    plt.plot(gamma_train, Y_train, '*', color='black', ms=10)
    plt.show()
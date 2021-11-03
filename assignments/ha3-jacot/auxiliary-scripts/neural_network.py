import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class FeedforwardNetwork(nn.Module):
    """Neural network implemented using NumPy."""

    def __init__(self, input_size, width=1000, beta=0.1, depth=4):
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N_eval = 100
    n_layers = 4
    beta = 0.1
    epochs = 1000
    width = 1000
    lr = 1
    seed = 1  # Change here for neural networks with different initializations

    # Train data
    gamma_obs = np.array([-2, -1.2, -0.4, 0.9, 1.8])
    X_obs = np.stack([np.cos(gamma_obs), np.sin(gamma_obs)]).T
    Y_obs = X_obs.prod(axis=1)

    # Test data
    gamma_eval = np.linspace(-np.pi, np.pi, N_eval)
    X_eval = np.stack([np.cos(gamma_eval), np.sin(gamma_eval)]).T
    Y_eval = X_eval.prod(axis=1)

    # Define neural network
    torch.manual_seed(seed)
    net = FeedforwardNetwork(2, beta=beta, depth=n_layers, width=width)

    # Convert tensors from numpy to pytorch
    X_obs_pth = torch.Tensor(X_obs)
    Y_obs_pth = torch.Tensor(Y_obs)
    X_eval_pth = torch.Tensor(X_eval)
    # Train
    msg = 'Epoch = {} - loss = {:0.2f}'
    pbar = tqdm.tqdm(initial=0, total=epochs, desc=msg.format(0, 0))
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for i in range(epochs):
        optimizer.zero_grad()
        Y_pred = net(X_obs_pth)
        error = (Y_obs_pth - Y_pred)
        loss = 1/2 * (error * error).mean()
        loss.backward()
        optimizer.step()
        # Update pbar
        pbar.desc = msg.format(i, loss)
        pbar.update(1)
    pbar.close()
    # Predict on test
    with torch.no_grad():
        Y_pred_eval = net(X_eval_pth).detach().numpy()

    # Plot on test data
    plt.plot(gamma_eval, Y_pred_eval, color='grey')
    plt.plot(gamma_eval, Y_eval, color='black')
    plt.plot(gamma_obs, Y_obs, '*', color='black', ms=10)
    plt.xlabel('gamma')
    plt.ylabel('f')
    plt.show()
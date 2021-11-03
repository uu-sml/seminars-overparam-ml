import numpy as np

# Small value it will be added to the covariance matrix to avoid numerical
# instabilities yielding a non positive definite covariacne matrix
EPS = 1e-7


def expkernel(l=1):
    def kernel(x1, x2):
        d = np.linalg.norm(x1 - x2, 2)
        out = np.exp(- d ** 2 / (2 * (l ** 2)))
        return out
    return kernel


def cov(x, kernel):
    """Receives a matrix (n_points, dimension) and return a kernel (n_points, n_points)"""
    n_points, dim = x.shape
    C = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            C[i, j] = kernel(x[i, :], x[j, :])
    return C


def conditioning(c, y_obs):
    n_obs = len(y_obs)
    if n_obs == 0:
        return np.zeros(c.shape[0]), c
    cov_obs = c[:n_obs, :n_obs]
    cov_eval_obs = c[n_obs:, :n_obs]
    cov_eval = c[n_obs:, n_obs:]

    inv_cov = np.linalg.inv(cov_obs)
    m = cov_eval_obs @ inv_cov @ y_obs
    cc = cov_eval - cov_eval_obs @ inv_cov @ cov_eval_obs.T
    return m, cc


if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt

    # Train data
    gamma_obs = np.array([-2, -1.2, -0.4, 0.9, 1.8])
    X_obs = np.stack([np.cos(gamma_obs), np.sin(gamma_obs)]).T
    Y_obs = X_obs.prod(axis=1)

    gamma_eval = np.linspace(-np.pi, np.pi, 100)
    X_eval = np.stack([np.cos(gamma_eval), np.sin(gamma_eval)]).T
    Y_eval = X_eval.prod(axis=1)

    # Generate plot before conditioning
    # Get covariance matrix
    c = cov(np.vstack([X_obs, X_eval]), kernel=expkernel(l=1))
    m, cc = conditioning(c, Y_obs)
    normal = multivariate_normal(mean=m.flatten(), cov=cc + EPS*np.eye(cc.shape[0]))
    Y_sols = normal.rvs(5)
    plt.plot(gamma_obs, Y_obs.T, '*', color=str(0.4), ms=10)
    plt.plot(gamma_eval, Y_eval.T, '-', color='black', ms=10)
    plt.plot(gamma_eval, Y_sols.T, color=str(0.4), ms=10)
    for nn in [3, 2, 1]:
        plt.fill_between(gamma_eval, m - nn*np.sqrt(np.diag(cc)),
                         m + nn*np.sqrt(np.diag(cc)), color=str(0.4+0.15 * nn), alpha=0.5)
    plt.xlabel('gamma')
    plt.ylabel('f')
    plt.show()

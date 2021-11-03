import numpy as np  #
import scipy.linalg as linalg


def train_and_evaluate(n_samples, n_features, n_test, noise_std, parameter_norm, seed,
                       datagen_parameter='gaussian-prior'):
    rng = np.random.RandomState(seed)

    # Get parameter
    if datagen_parameter == 'gaussian-prior':
        beta = parameter_norm / np.sqrt(n_features) * rng.randn(n_features)
    elif datagen_parameter == 'constant':
        beta = parameter_norm / np.sqrt(n_features) * np.ones(n_features)

    # Generate training data
    # Get X matrix
    X = rng.randn(n_samples, n_features)
    # Get error
    e = rng.randn(n_samples)
    # Compute output
    y = X @ beta + noise_std * e

    # Train
    beta_hat, _resid, _rank, _s = linalg.lstsq(X, y)

    # Test data
    # Get X matrix
    X_test = rng.randn(n_test, n_features)
    # Get error
    e_test = rng.randn(n_test)
    # Compute output
    y_test = X_test @ beta + noise_std * e_test

    # Computer test error and parameter loss
    test_error = np.mean((X_test @ beta_hat - y_test)**2)
    parameter_norm = np.linalg.norm(beta_hat, ord=2)

    return test_error, parameter_norm


def asymptotic_risk(proportion, signal_amplitude, noise_std):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix

    # The variance term
    v_underparametrized = proportion / (1 - proportion)
    v_overparametrized = 1 / (proportion - 1)
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    # The bias term
    b_underparametrized = 0
    b_overparametrized = (1 - 1 / proportion)
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b + noise_std ** 2


def assymptotic_l2_norm(proportion, signal_amplitude, noise_std):
    v_underparametrized = proportion / (1 - proportion)
    v_overparametrized = 1 / (proportion - 1)
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 1
    b_overparametrized = 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return np.sqrt(noise_std ** 2 * v + signal_amplitude ** 2 * b)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_experiments = 100
    n_pts_asymp = 1000
    n_train = 200
    n_test = 200
    noise_std = 0.0
    parameter_norm = 1.0
    seed = 0

    # empirical evaluation
    n_features = np.zeros(n_experiments, dtype=int)
    mse_test = np.zeros(n_experiments)
    param_norm = np.zeros(n_experiments)
    # Repeat experiments for different number of parameters
    for i, fraction in enumerate(np.logspace(-1, 1, n_experiments)):
        # initialize and train model
        n_features[i] = int(fraction * n_train)
        # train and evaluate
        mse_test[i], param_norm[i] = train_and_evaluate(n_train, n_features[i], n_test, noise_std, parameter_norm, seed)

    # Asymptotics
    proportion = np.logspace(-1, 1, n_pts_asymp)
    a_mse = asymptotic_risk(proportion, parameter_norm, noise_std)
    a_norm = assymptotic_l2_norm(proportion, parameter_norm, noise_std)

    fig, ax = plt.subplots(2)
    l, = ax[0].plot(n_features / n_train, mse_test, marker='*', ls='')
    ax[0].plot(proportion, a_mse, ls='-', color=l.get_color())
    ax[0].set_ylim([0, 10])
    ax[0].axvline(1, ls='--')
    ax[0].set_ylabel('Mean square error')
    ax[0].set_xscale('log')
    ax[0].legend()
    l, = ax[1].plot(n_features / n_train, param_norm, marker='*', ls='')
    ax[1].plot(proportion, a_norm, ls='-', color=l.get_color())
    ax[1].set_ylabel('Parameter norm')
    ax[1].set_xlabel('# features / # datapoints')
    ax[1].set_xscale('log')
    ax[1].axvline(1, ls='--')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def arrowed_spines(ax=None, arrowLength=30, labels=('X', 'Y'), arrowStyle='<|-'):
    xlabel, ylabel = labels

    for i, spine in enumerate(['left', 'bottom']):
        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [1, 0], ('axes fraction', t)
        xytext, textcoords = [arrowLength, 0], ('offset points', t)

        # create arrowprops
        arrowprops = dict( arrowstyle=arrowStyle,
                           facecolor=ax.spines[spine].get_facecolor(),
                           linewidth=ax.spines[spine].get_linewidth(),
                           alpha = ax.spines[spine].get_alpha(),
                           zorder=ax.spines[spine].get_zorder(),
                           linestyle = ax.spines[spine].get_linestyle() )

        if spine is 'bottom':
            ha, va = 'left', 'center'
            xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext,
                        textcoords=textcoords, ha=ha, va='center',
                        arrowprops=arrowprops)
        else:
            ha, va = 'center', 'bottom'
            yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1],
                        xytext=xytext[::-1], textcoords=textcoords[::-1],
                        ha='center', va=va, arrowprops=arrowprops)
    return xarrow, yarrow


if __name__ == "__main__":
    N_test = 100

    # Train data
    gamma_train = np.array([-2, -1.2, -0.4, 0.9, 1.8])
    X_train = np.stack([np.cos(gamma_train), np.sin(gamma_train)]).T
    Y_train = X_train.prod(axis=1)

    # Test data
    gamma_test = np.linspace(-np.pi, np.pi, N_test)
    X_test = np.stack([np.cos(gamma_test), np.sin(gamma_test)]).T
    Y_test = X_test.prod(axis=1)

    # Plot dataset
    plt.style.use('./mystyle.mplsty')
    plt.plot(gamma_train, Y_train, 'o', color='black', ms=10, label="$f(x_i)$")
    plt.plot(gamma_test, Y_test, '-', color='black', ms=10)
    plt.xlabel("$\gamma$")
    plt.ylabel("$f(\cos\,\gamma, \sin\,\gamma)$")
    plt.legend()
    plt.savefig('function1d.pdf')

    plt.style.use('./mystyle2.mplsty')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_train[:, 0], X_train[:, 1], 'o', color='black', ms=10, label="$f(x_i)$")
    ax.plot(X_test[:, 0], X_test[:, 1], '-', color='black', ms=10, label="$f(x_i)$")
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim([-3, 3])
    ax.set_xlim([-4, 4])
    arrowed_spines(ax, labels=('$x_{i1}$', '$x_{i2}$'))
    plt.savefig('function2d.pdf')
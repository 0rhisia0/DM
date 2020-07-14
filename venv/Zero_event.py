import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import constants as const
import scipy.stats as stats

def main():
    ybins = 10 ** np.linspace(-48, -40, 200)
    xbins = 10 ** np.linspace(0, 3, 200)
    print(stats.poisson.pmf(k=0, mu=1))
    plt.plot(xbins, ybins)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


if __name__ == "__main__":
    main()
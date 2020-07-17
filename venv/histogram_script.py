import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import constants as const
import scipy.stats as stats
from scipy.ndimage import gaussian_filter


def find_credible(H, alpha, level):
    B = np.copy(H)
    norm = np.sum(B)
    run_tot = 0
    list = []
    while run_tot < norm * (1-alpha):
        coord = np.unravel_index(np.argmax(B), B.shape)
        list.append(coord)
        run_tot += B[coord]
        B[coord] = 0
    list = np.asarray(list)
    new = np.zeros(B.shape)
    for i in list:
        new[i[0], i[1]] = level
    return new


def main():
    with open('data.txt') as f:
        thetas = f.read().splitlines()
    for i in range(len(thetas)):
        thetas[i] = thetas[i][1:-1].split()
        for j in range(2):
            thetas[i] = [float(i) for i in thetas[i]]
    thetas = np.asarray(thetas)
    thetas = thetas[:]
    ybins = 10 ** np.linspace(-50, -40, 200)
    xbins = 10 ** np.linspace(0, 3, 200)

    fig, ax = plt.subplots()
    h = ax.hist2d(thetas[:, 0]/const.GeV, thetas[:, 1]/const.cm2, bins=[xbins, ybins], norm=mcolors.PowerNorm(0.4))
    plt.colorbar(h[3], ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.plot(300*const.GeV/10**6, const.sigma/(10*const.cm2), marker="x", color="r")
    ax.set_xlabel(r"Mass ($GeV$)")
    ax.set_ylabel(r'$\sigma_n$ ($cm^2$)')
    plt.show()

    fig, ax = plt.subplots()
    H, xedges, yedges = np.histogram2d(thetas[:, 0]/const.GeV, thetas[:, 1]/const.cm2, bins=[xbins, ybins])
    H = H.T

    cr_2 = find_credible(H, 0.05, 1)
    cr_2 = gaussian_filter(cr_2, 1, mode='constant')
    cr_2 = np.around(cr_2)
    # cr_1 = find_credible(H, 0.43, 1)
    # cr_1 = gaussian_filter(cr_1, 0.5, mode='constant')
    cmap = plt.get_cmap("binary")
    X, Y = np.meshgrid(xedges, yedges)
    ax.contourf(X[:-1, :-1], Y[:-1, :-1], cr_2, alpha=0.3, cmap=cmap)
    # ax.contourf(X[:-1, :-1], Y[:-1, :-1], cr_1, alpha=0.3, cmap=cmap)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.plot(300*const.GeV/10**6, const.sigma/(10*const.cm2), marker="x", color="r")
    ax.set_xlabel(r"Mass ($GeV$)")
    ax.set_ylabel(r'$\sigma_n$ ($cm^2$)')
    plt.show()



if __name__ == "__main__":
    main()
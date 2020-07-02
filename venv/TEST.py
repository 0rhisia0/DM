import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Likelihood as lik
import constants as const
import matplotlib.colors as mcolors
from tqdm import tqdm
from math import isnan
M_D_Low = 1 * const.GeV
M_D_High = 1000 * const.GeV
sigma_high = 1e-38 * const.cm2
sigma_low = 1e-48 * const.cm2

def generate_data(N):
    masses = stats.norm.rvs(100*const.GeV, 30*const.GeV, size=500)
    sigmas = stats.lognorm.rvs(1, scale=1e-45*const.cm2, loc=0, size=500)
    ybins = 10 ** np.linspace(-48, -43, 50)
    xbins = np.linspace(1*const.GeV, 1100*const.GeV, 50)
    plt.hist2d(masses, sigmas/const.cm2, bins=[xbins, ybins])
    plt.yscale("log")
    plt.show()
    return masses, sigmas


def proposal(WIMP_curr):
    # build a gaussian curve to sample from in the parameter space
    # uses a normal in mass space and a log normal in cross section space to sample the region
    curr_mass = WIMP_curr[0]
    curr_sigma = WIMP_curr[1]
    M_D_new = 0
    sigma_new = 0
    while True:
        M_D_new = np.random.normal(curr_mass, 20 * const.GeV)  # sample param for mass
        if M_D_Low <= M_D_new <= M_D_High:
            break
    while True:
        sigma_new = stats.lognorm.rvs(0.1, scale=curr_sigma, loc=0)  # sample param for cross section
        if sigma_low <= sigma_new <= sigma_high:
            break
    assert (M_D_new and sigma_new)
    return M_D_new, sigma_new


def likelihood(masses, sigmas, mass, sigma):
    prob = np.sum(np.log(stats.norm.pdf(masses, loc=mass, scale=10*const.GeV)))
    prob += np.sum(np.log(stats.lognorm.pdf(sigmas, s=1, scale=sigma, loc=0)))
    return prob


def main():
    N = 10000
    masses, sigmas = generate_data(100)
    curr_mass = 100*const.GeV
    curr_sigma = 1e-39*const.cm2
    post_mass = []
    post_sigma = []
    acceptance = []
    for i in tqdm(range(N)):
        new_mass, new_sigma = proposal([curr_mass, curr_sigma])
        prev_lik = likelihood(masses, sigmas, curr_mass, curr_sigma)
        new_lik = likelihood(masses, sigmas, new_mass, new_sigma)
        ratio = new_lik-prev_lik
        if isnan(ratio):
            print("JUMP")
            curr_mass = stats.uniform.rvs(loc=M_D_Low, scale=M_D_High-M_D_Low)
            curr_sigma = stats.uniform.rvs(loc=sigma_low, scale=sigma_high-sigma_low)
            continue
        elif not i % 4:
            acceptance.append(1)
            curr_mass = new_mass
            curr_sigma = new_sigma
        elif ratio >= 0:
            acceptance.append(1)
            curr_mass = new_mass
            curr_sigma = new_sigma
        elif np.log10(np.random.rand()) < ratio:
            acceptance.append(1)
            curr_mass = new_mass
            curr_sigma = new_sigma
        post_mass.append(curr_mass)
        post_sigma.append(curr_sigma)
    move_percent = len(acceptance)/N
    print(move_percent)
    fig, ax = plt.subplots()
    ybins = 10 ** np.linspace(-48, -43, 50)
    xbins = np.linspace(1*const.GeV, 1100*const.GeV, 50)
    h = ax.hist2d(post_mass, np.asarray(post_sigma)/const.cm2, bins=[xbins, ybins], norm=mcolors.PowerNorm(0.4))
    plt.yscale("log")
    plt.colorbar(h[3], ax=ax)
    ax.plot(post_mass, np.asarray(post_sigma)/const.cm2, alpha=0.3, color='r')
    plt.xlabel("Mass of WIMP (GeV)")
    plt.ylabel("Cross section of WIMP (cm^2)")
    plt.show()


if __name__ == "__main__":
    main()

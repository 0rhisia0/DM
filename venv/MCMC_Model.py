import constants as const
import numpy as np
from scipy.stats import lognorm
from scipy import integrate
from matplotlib import pyplot as plt
import Likelihood as lik
import functions as fn
import matplotlib as mpl

# given events we want to find the parameters that match them best
# We can do this using a Markov Chain Monte Carlo utilizing the Metropolis Hastings algorithm
M_D_Low = 1 * const.GeV
M_D_High = 300 * const.GeV
sigma_high = 1e-43 * const.cm2
sigma_low = 1e-48 * const.cm2
N = 1000  # number of markov chain transitions


def proposal(WIMP_curr):
    # build a gaussian curve to sample from in the parameter space
    # uses a normal in mass space and a log normal in cross section space to sample the region
    curr_mass = WIMP_curr[0]
    curr_sigma = WIMP_curr[1]
    M_D_new = 0
    sigma_new = 0
    while True:
        M_D_new = np.random.normal(curr_mass, 30 * const.GeV)  # sample param for mass
        if M_D_Low <= M_D_new <= M_D_High:
            break
    while True:
        sigma_new = lognorm.rvs(0.8, scale=curr_sigma, loc=0)  # sample param for cross section
        if sigma_low <= sigma_new <= sigma_high:
            break
    assert (M_D_new and sigma_new)
    return np.asarray([M_D_new, sigma_new])


def main():
    # # N = 1000
    #
    # events = []
    # with open('mock.txt') as f:
    #     events = f.read().splitlines()
    # events = [float(num) for num in events]
    # # define events
    # E_thr = 6 * const.keV
    # # pick start
    # theta = np.zeros((N, 2))
    # # initialize MCMC parameters
    # theta[0] = [const.M_D, const.sigma]
    # # initialize values for energy
    # Emin = 0.001 * const.keV
    # Emax = fn.max_recoil_energy()
    # Nsteps = 100
    # del_Er = (Emax - Emin) / Nsteps
    # E_r = np.arange(Emin, Emax, del_Er)
    # # Metropolis Hastings loop
    # for i in range(1, N):
    #     proposed_theta = proposal(theta[i - 1])
    #     numer = lik.events_likelihood(E_r, events, theta[i - 1], const.AXe, E_thr, del_Er)
    #     denom = lik.events_likelihood(E_r, events, proposed_theta, const.AXe, E_thr, del_Er)
    #     if denom == 0:
    #         ratio = 1
    #     else:
    #         ratio = numer / denom
    #     if ratio == 0: print("yes")
    #     comp = np.random.uniform(0, 1)
    #     if comp >= np.minimum(ratio, 1):
    #         theta[i] = proposed_theta
    #     else:
    #         theta[i] = theta[i - 1]
    # ybins = 10 ** np.linspace(-48, -43, 50)
    # xbins = np.linspace(M_D_Low, M_D_High, 50)
    #
    # fig, ax = plt.subplots()
    # ax.hist2d(theta[:, 0], theta[:, 1] / const.cm2, bins=[xbins, ybins])
    # ax.set_yscale('log')
    # plt.show()

    theta = np.zeros((N, 2))
    theta[0] = np.asarray([const.M_D, const.sigma])
    for i in range(1, N):
        theta[i] = proposal(theta[i - 1])
    print(theta)
    x = theta[:, 0]
    y = theta[:, 1]
    plt.scatter(x, y, alpha=0.2)
    plt.yscale('log')
    plt.ylim(1e-52, 1e-47)
    plt.show()


if __name__ == "__main__":
    main()

import constants as const
import numpy as np
from scipy import stats
from scipy.stats import lognorm
from scipy import integrate
from matplotlib import pyplot as plt
import Likelihood as lik
import functions as fn
import matplotlib.colors as mcolors
from tqdm import tqdm
from math import isnan, isinf
import MockGen


# given events we want to find the parameters that match them best
# We can do this using a Markov Chain Monte Carlo utilizing the Metropolis Hastings algorithm
M_D_Low = 1 * const.GeV
M_D_High = 1000 * const.GeV
sigma_high = 1e-40 * const.cm2
sigma_low = 1e-48 * const.cm2
N = 10000  # number of markov chain transitions


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
        sigma_new = lognorm.rvs(0.2, scale=curr_sigma, loc=0)  # sample param for cross section
        if sigma_low <= sigma_new <= sigma_high:
            break
    assert (M_D_new and sigma_new)
    return np.asarray([M_D_new, sigma_new])


def main():
    TRUE_WIMP = MockGen.TRUE_WIMP
    # N = 1000
    print()
    events = []
    with open('mock.txt') as f:
        events = f.read().splitlines()
    events = [float(num) for num in events]
    # define events
    E_thr = 6 * const.keV
    # pick start
    theta = np.zeros((N, 2))
    # initialize MCMC parameters
    theta[0] = [3*const.M_D, 100*const.sigma]
    # initialize values for energy
    Emin = 1 * const.keV
    Emax = fn.max_recoil_energy()
    Nsteps = 159
    del_Er = 1
    E_r = np.arange(Emin, Emax, del_Er)
    events = lik.find_indices(E_r, events)
    # Metropolis Hastings loop
    hooked = False
    acceptance = []
    for i in tqdm(range(1, N)):
        proposed_theta = proposal(theta[i - 1])
        prev_lik = lik.events_likelihood(E_r, events, theta[i - 1], const.AXe, E_thr, del_Er)
        new_lik = lik.events_likelihood(E_r, events, proposed_theta, const.AXe, E_thr, del_Er)
        ratio = new_lik - prev_lik
        if ratio >= 0:
            theta[i] = proposed_theta
            acceptance.append(1)
        elif i%3==0:
            theta[i] = proposed_theta
            acceptance.append(1)
        else:
            theta[i] = theta[i-1]
    print(len(acceptance)/10000)
    ybins = 10 ** np.linspace(-48, -40, 50)
    xbins = 10 ** np.linspace(6, 9, 50)
    fig, ax = plt.subplots()
    ax.hist2d(theta[:, 0], theta[:, 1] / const.cm2, bins=[xbins, ybins])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(TRUE_WIMP[0], TRUE_WIMP[1]/const.cm2, marker="x", color="r")
    ax.set_xlabel("Mass of WIMP (eV)")
    ax.set_ylabel("Cross section of WIMP (cm^2)")
    plt.show()
    print(N-1)
    #
    # theta = np.zeros((N, 2))
    # theta[0] = np.asarray([const.M_D, const.sigma])
    # for i in range(1, N):
    #     theta[i] = proposal(theta[i - 1])
    # print(theta)
    # x = theta[:, 0]
    # y = theta[:, 1]/const.cm2
    # plt.plot(x, y)
    # plt.yscale('log')
    # plt.ylim(1e-48, 1e-40)
    # plt.show()


if __name__ == "__main__":
    main()

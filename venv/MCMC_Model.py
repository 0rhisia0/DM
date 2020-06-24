import constants as const
import numpy as np
from scipy.stats import lognorm
from scipy import integrate
from matplotlib import pyplot as plt
import Likelihood as lik

# given events we want to find the parameters that match them best
# We can do this using a Markov Chain Monte Carlo utilizing the Metropolis Hastings algorithm
M_D_Low = 1*const.GeV
M_D_High = 300*const.GeV
sigma_high = 1e-43*const.cm2
sigma_low = 1e-48*const.cm2
N = 10  # number of markov chain transitions


def proposal(WIMP_curr):
    # build a gaussian curve to sample from in the parameter space
    # uses a normal in mass space and a log normal in cross section space to sample the region
    curr_mass = WIMP_curr[0]
    curr_sigma = WIMP_curr[1]
    M_D_new = 0
    sigma_new = 0
    while True:
        M_D_new = np.random.normal(curr_mass, 30*const.GeV)  # sample param for mass
        if M_D_Low <= M_D_new <= M_D_High:
            break
    while True:
        sigma_new = lognorm.rvs(0.5, scale=curr_sigma, loc=0)  # sample param for cross section
        if sigma_low <= sigma_new <= sigma_high:
            break
    assert (M_D_new and sigma_new)
    return np.asarray([M_D_new, sigma_new])


def main():
    N = 1000
    theta = np.zeros((N, 2))
    theta[0] = np.asarray([const.M_D, const.sigma])
    for i in range(1, N):
        theta[i] = proposal(theta[i - 1])
    print(theta)
    plt.plot(theta[:, 0], theta[:, 1]/const.cm2, alpha=0.5)
    plt.yscale('log')
    plt.ylim(10**-49, 10**-42)
    plt.show()


    # # define events
    # events = []
    # # pick start
    # theta = np.zeros((N, 2))
    # # initialize MCMC parameters
    # theta[0] = [const.M_D, const.sigma]
    # #Metropolis Hastings loop
    # for i in range(1, N):
    #     proposed_theta = proposal(theta[i-1])
    #     numer = lik.events_likelihood(0.001*const.keV, 200*const.keV, 1000,
    #                               events, theta[i-1], const.AXe, 10*const.keV)
    #     denom = lik.events_likelihood(0.001*const.keV, 200*const.keV, 1000,
    #                               events, proposed_theta, const.AXe, 10*const.keV)
    #     ratio = numer/denom
    #     comp = np.random.uniform(0, 1)
    #     if comp >= np.min([1, ratio]):
    #         theta[i] = proposed_theta
    #     else:
    #         theta[i] = theta[i-1]
    # print(theta)

if __name__ == "__main__":
    main()

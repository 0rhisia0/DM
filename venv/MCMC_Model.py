import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt
import Likelihood as lik

# given events we want to find the parameters that match them best
# We can do this using a Markov Chain Monte Carlo utilizing the Metropolis Hastings algorithm
M_D_Low = 1*const.Gev
M_D_Low = 3*const.Gev
sigma_high = 10e-47
sigma_low = 10e-43
N = 200000  # number of markov chain transitions

if __name__=="__main__":
    main()

def proposal(WIMP_curr):
    M_D_new = sp.stats.norm() #sample params
    sigma_new = sp.stats.norm() #sample params
    return WIMP_new

def main():
    #define events
    events = [0]
    #pick start
    theta = np.zeros(N)
    #initialize MCMC search
    theta[0] = [const.M_D, const.sigma]
    for i in range(1, N):
        theta[i] = proposal(theta[i-1])
        numer = lik.events_likelihood(0.001*const.keV, 200*const.keV, 1000,
                                  events, theta[i-1], const.AXe, 10*const.keV)
        denom = lik.events_likelihood(0.001*const.keV, 200*const.keV, 1000,
                                  events, theta[i], const.AXe, 10*const.keV)
        ratio = numer/denom
        comp = np.random(0, 1)
        if comp >= np.min([1,ratio]):
            theta[i] = theta[i-1]
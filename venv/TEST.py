import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Likelihood as lik
import constants as const
import matplotlib as mpl
M_D_Low = 1 * const.GeV
M_D_High = 300 * const.GeV
sigma_high = 1e-43 * const.cm2
sigma_low = 1e-48 * const.cm2

def generate_data(N):
    masses = stats.norm.rvs(const.M_D*2.1, 10*const.GeV, size=500)
    sigmas = stats.lognorm.rvs(1, scale=const.sigma*10, loc=0, size=500)
    ybins = 10 ** np.linspace(-48, -43, 30)
    xbins = np.linspace(1*const.GeV, 300*const.GeV, 30)
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
        M_D_new = np.random.normal(curr_mass, 5 * const.GeV)  # sample param for mass
        if M_D_Low <= M_D_new <= M_D_High:
            break
    while True:
        sigma_new = stats.lognorm.rvs(0.3, scale=curr_sigma, loc=0)  # sample param for cross section
        if sigma_low <= sigma_new <= sigma_high:
            break
    assert (M_D_new and sigma_new)
    return M_D_new, sigma_new


def likelihood(masses, sigmas, mass, sigma):
    prob = 0
    for i in range(len(masses)):
        prob += np.log10(stats.norm.pdf(masses[i], loc=mass, scale=10*const.GeV))
        prob += np.log10(stats.lognorm.pdf(sigmas[i], s=1, scale=sigma, loc=0))
    return prob

def main():
    masses, sigmas = generate_data(100)
    curr_mass = const.M_D*3
    curr_sigma = const.sigma*10
    post_mass = [curr_mass]
    post_sigma = [curr_sigma]
    acceptance = np.zeros(1000)
    for i in range(1000):
        new_mass, new_sigma = proposal([curr_mass, curr_sigma])
        prev_lik = likelihood(masses, sigmas, curr_mass, curr_sigma)
        new_lik = likelihood(masses, sigmas, new_mass, new_sigma)
        ratio = 10**(new_lik-prev_lik)
        if ratio > 1:
            acceptance[i] = 1
            curr_mass = new_mass
            curr_sigma = new_sigma
        elif np.random.rand() < ratio:
            acceptance[i] = 1
            curr_mass = new_mass
            curr_sigma = new_sigma
        else:
            acceptance[i] = 0
        post_mass.append(curr_mass)
        post_sigma.append(curr_sigma)
    move_percent = len(acceptance[acceptance == 1])/len(acceptance)
    print(move_percent)
    print(post_sigma[100], post_mass[100])
    ybins = 10 ** np.linspace(-48, -43, 50)
    xbins = np.linspace(1*const.GeV, 300*const.GeV, 50)
    plt.hist2d(post_mass, np.asarray(post_sigma)/const.cm2, bins=[xbins, ybins], norm=mpl.colors.LogNorm(), cmap=mpl.cm.gray)
    plt.yscale("log")
    plt.show()
    plt.plot(post_mass, np.asarray(post_sigma)/const.cm2)
    plt.yscale("log")
    plt.show()



if __name__=="__main__":
    main()
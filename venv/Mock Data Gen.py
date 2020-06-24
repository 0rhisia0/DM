import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import stats
from scipy import integrate
from matplotlib import pyplot as plt
Nsteps = 10000

def main():
    WIMP = [const.M_D, const.sigma]
    E_min = 0.001*const.keV
    E_max = fn.max_recoil_energy()
    del_Er = (E_max - E_min) / Nsteps
    E_r = np.arange(E_min, E_max, del_Er)
    x, y = fn.integrate_rate(E_r, WIMP, const.AXe)
    norm_fact = np.sum(y)
    print(norm_fact)
    y = y/norm_fact
    # plt.plot(x, y)
    # plt.yscale("log")
    # plt.show()
    print(np.sum(y))
    custm = stats.rv_discrete(name='custm', values=(x, y))
    fig, ax = plt.subplots(1, 1)
    samples = custm.rvs(size=1000000)
    ax.hist(samples, bins=100)
    plt.yscale('log')
    plt.show()




if __name__=="__main__":
    main()
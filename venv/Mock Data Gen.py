import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import Likelihood as lik
N_STEPS = 100  # Energy steps

def main():
    E_thr = 6*const.keV  # threshold energy for generating total number of events
    WIMP = [const.M_D, const.sigma]  # WIMP params

    target = const.AXe
    E_min = 0.001*const.keV
    E_max = fn.max_recoil_energy()
    del_Er = (E_max - E_min) / N_STEPS
    E_r = np.arange(E_min, E_max, del_Er)
    x, y = fn.integrate_rate(E_r, WIMP, target)  # expected events per kg, as a function of E_thr
    y *= lik.BULK  # 100 kilo target, 300 day runtime

    idx = lik.find_nearest_idx(E_r, E_thr)
    mean_events = y[idx]  # expected number of events
    num_events = stats.poisson.rvs(mean_events)  # num events to generate

    event_dist = fn.diff_rate(E_r, WIMP, target)  # event energies distribution

    norm_fact = np.sum(event_dist)
    event_dist /= norm_fact

    custm = stats.rv_discrete(name='custm', values=(E_r, event_dist))

    samples = custm.rvs(size=num_events)
    samples = samples[E_thr < samples]
    samples = samples[samples < 100]
    plt.hist(samples, bins=100)
    plt.yscale('log')
    plt.show()
    f = open('mock.txt', 'w')
    for ele in samples:
        f.write(str(ele) + '\n')
    f.close()



if __name__=="__main__":
    main()
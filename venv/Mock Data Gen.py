import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import Likelihood as lik
N_STEPS = 159  # Energy steps

def main():
    E_thr = 6*const.keV  # threshold energy for generating total number of events
    WIMP = [const.M_D, const.sigma]  # WIMP params

    target = const.AXe
    E_min = 1*const.keV
    E_max = fn.max_recoil_energy()
    del_Er = (E_max - E_min) / N_STEPS
    E_r = np.arange(E_min, E_max, del_Er)
    x, y = fn.integrate_rate(E_r, WIMP, target)  # expected events per kg day, as a function of E_thr
    y *= lik.BULK  # 100 kilo target, 300 day runtime

    idx = lik.find_nearest_idx(E_r, E_thr)
    mean_events = y[idx]  # expected number of events
    num_events = stats.poisson.rvs(np.around(mean_events))  # num events to generate

    event_dist = fn.diff_rate(E_r, WIMP, target)  # event energies distribution
    idx2 = lik.find_nearest_idx(E_r, 60*const.keV)
    event_dist = event_dist[idx:idx2]
    E_r = E_r[idx:idx2]

    norm_fact = np.sum(event_dist)
    event_dist /= norm_fact
    true_rates = event_dist*mean_events

    custm = stats.rv_discrete(name='custm', values=(E_r, event_dist))

    samples = custm.rvs(size=num_events)
    samples = samples[E_thr < samples]
    samples = samples[samples < 100]
    fig, ax = plt.subplots()
    ax.hist(samples, bins=N_STEPS)
    for WIMP in [[const.M_D, const.sigma], [const.M_D*0.5, const.sigma], [const.M_D*2, const.sigma]]:
        E_min = 1 * const.keV
        E_max = fn.max_recoil_energy()
        del_Er = (E_max - E_min) / N_STEPS
        E_r = np.arange(E_min, E_max, del_Er)

        event_dist = fn.diff_rate(E_r, WIMP, target)  # event energies distribution
        event_dist = event_dist[idx:60]
        E_r = E_r[idx:60]

        norm_fact = np.sum(event_dist)
        event_dist /= norm_fact
        true_rates = event_dist * mean_events
        ax.plot(E_r, true_rates, label=str(WIMP))
    plt.legend()
    plt.show()
    f = open('mock.txt', 'w')
    for ele in samples:
        f.write(str(ele) + '\n')
    f.close()



if __name__=="__main__":
    main()
import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import Likelihood as lik
N_STEPS = 159  # Energy steps
TRUE_WIMP = [300*const.GeV, const.sigma*10]

def main():
    E_thr = 6*const.keV  # threshold energy for generating total number of events
    WIMP = TRUE_WIMP  # WIMP params

    target = const.AXe
    E_min = 1*const.keV
    E_max = fn.max_recoil_energy()
    del_Er = (E_max - E_min) / N_STEPS
    E_r = np.arange(E_min, E_max, del_Er)
    weights = 1/(1+np.exp(-1.35*E_r+8.1))

    x, y = fn.integrate_rate(E_r, WIMP, target)  # expected events per kg day, as a function of E_thr
    y *= lik.BULK  # 100 kilo target, 300 day runtime

    idx = lik.find_nearest_idx(E_r, E_thr)
    mean_events = y[idx]  # expected number of events
    num_events = stats.poisson.rvs(mean_events)  # num events to generate

    event_dist = fn.diff_rate(E_r, WIMP, target)  # event energies distribution
    idx2 = lik.find_nearest_idx(E_r, 60*const.keV)
    event_dist *= weights
    event_dist = event_dist[:idx2]
    E_r = E_r[:idx2]

    norm_fact = np.sum(event_dist)
    event_dist /= norm_fact
    true_rates = event_dist*mean_events

    custm = stats.rv_discrete(name='custm', values=(E_r, event_dist))

    samples = custm.rvs(size=num_events)
    samples = samples[samples < 100]
    fig, ax = plt.subplots()
    ax.hist(samples, bins=N_STEPS)
    for WIMP in [TRUE_WIMP, [500*const.GeV, TRUE_WIMP[1]], [const.M_D*10, TRUE_WIMP[1]]]:
        E_min = 1 * const.keV
        E_max = fn.max_recoil_energy()
        del_Er = (E_max - E_min) / N_STEPS
        E_r = np.arange(E_min, E_max, del_Er)

        event_dist = fn.diff_rate(E_r, WIMP, target)  # event energies distribution
        event_dist *= weights
        event_dist = event_dist[:60]
        E_r = E_r[:60]

        norm_fact = np.sum(event_dist)
        event_dist /= norm_fact
        true_rates = event_dist * mean_events
        ax.plot(E_r, true_rates, label=str(WIMP[0]/10**6)+" GeV, "+str(WIMP[1]/const.cm2)+"cm2" )
    ax.set_ylabel("Counts")
    ax.set_xlabel("Recoil Energy (keV)")
    plt.legend()
    plt.show()
    f = open('mock.txt', 'w')
    for ele in samples:
        f.write(str(ele) + '\n')
    f.close()



if __name__=="__main__":
    main()
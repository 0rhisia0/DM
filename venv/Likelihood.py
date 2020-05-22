import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def events_likelihood(e_min, e_max, n_steps, events, WIMP, A, E_thr):
    del_Er = (e_max - e_min) / n_steps
    Er = np.arange(e_min, e_max, del_Er)
    likelihood = pois_and_en_prob(events, WIMP, A, Er, E_thr)
    return likelihood


def pois_and_en_prob(events, WIMP, A, Er, E_thr):
    p_energies = 1
    dif_rate = fn.diff_rate2(Er, WIMP, A)
    Er, int_rate = fn.integrate_rate2(Er, WIMP, A)
    p_d_f_energies = dif_rate / int_rate
    for event_energy in events:
        idx = find_nearest_idx(event_energy, Er)
        p_energies *= p_d_f_energies[idx]
    obs_events = len(events)
    pred_events = np.around(int_rate[find_nearest_idx(E_thr, Er)])
    likelihood = obs_events*np.exp(pred_events)*p_energies/np.math.factorial(pred_events)
    return likelihood


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main():
    Emin = 0.001 * const.keV
    Emax = 200. * const.keV
    Nsteps = 10000
    del_Er = (Emax - Emin) / Nsteps
    Er = np.arange(Emin, Emax, del_Er)
    print(events_likelihood(0.001*const.keV, 200*const.keV, 1000
                            , [15*const.keV], [const.M_D, const.sigma], const.AXe, 10*const.keV))

if __name__=="__main__":
    main()
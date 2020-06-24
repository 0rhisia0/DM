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
    dif_rate = fn.diff_rate(Er, WIMP, A)
    Er, int_rate = fn.integrate_rate(Er, WIMP, A)
    print(find_nearest_idx(int_rate, 0))
    assert (0 not in dif_rate)
    assert (0 not in int_rate), "0 in int rate"
    p_d_f_energies = dif_rate/int_rate
    for event_energy in events:
        idx = find_nearest_idx(event_energy, Er)
        p_energies *= p_d_f_energies[idx]
    obs_events = len(events)
    pred_events = np.around(int_rate[find_nearest_idx(E_thr, Er)])
    likelihood = obs_events * np.exp(pred_events) * p_energies / np.math.factorial(pred_events)
    return likelihood


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main():
    Emin = 0.001 * const.keV
    Emax = fn.max_recoil_energy()
    Nsteps = 1000
    del_Er = (Emax - Emin) / Nsteps
    Er = np.arange(Emin, Emax, del_Er)
    print(events_likelihood(0.001 * const.keV, Emax, 1000
                            , [10, 12, 55, 33, 122, 12, 14, 17,10, 12, 55, 33, 122, 12, 14, 17], [const.M_D, const.sigma], const.AXe, 10 * const.keV))


if __name__ == "__main__":
    main()

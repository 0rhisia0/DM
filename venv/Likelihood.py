import functions as fn
import constants as const
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy import integrate

BULK = 100 * 30


def events_likelihood(E_r, events, WIMP, A, E_thr, del_Er):
    p_energies = 1

    obs_events = len(events)  # num events observed
    idx = find_nearest_idx(E_r, E_thr)
    E_r, int_rate = fn.integrate_rate(E_r, WIMP, A)
    norm_fact = int_rate[idx]

    # normalization factor of energy probabilities
    int_rate = int_rate[idx:100] * BULK

    idx = find_nearest_idx(E_r, E_thr)
    pred_events = np.around(int_rate[idx])

    poisson_prob = poisson.pmf(obs_events, pred_events)
    e_prob = energy_prob(events, WIMP, A, norm_fact, E_r, del_Er)
    return poisson_prob


def energy_prob(events, WIMP, A, norm_fact, E_r, del_Er):
    prob = 1
    dif_rate = fn.diff_rate(E_r, WIMP, A)
    p_m_f = dif_rate / norm_fact
    print(integrate.simps(p_m_f, E_r), end="\r")
    for event in events:
        idx = find_nearest_idx(E_r, event)
        prob *= p_m_f[idx]*del_Er
    return prob


def error(dif_rate, E_r, WIMP):
    print(WIMP)
    plt.plot(E_r, dif_rate)
    plt.yscale('log')
    plt.show()


def find_nearest_idx(array, value):
    array = array - value
    idx = (np.abs(array)).argmin()
    return idx


def main():
    Emin = 0.001 * const.keV
    Emax = fn.max_recoil_energy()
    Nsteps = 200
    del_Er = (Emax - Emin) / Nsteps
    E_r = np.arange(Emin, Emax, del_Er)
    with open('mock.txt') as f:
        events = f.read().splitlines()
    events = [float(num) for num in events]
    WIMP = [const.M_D, const.sigma]
    nucleus = const.AXe
    E_thr = 6 * const.keV
    idx = find_nearest_idx(E_r, E_thr)
    E_r, int_rate = fn.integrate_rate(E_r, WIMP, nucleus)
    norm_fact = int_rate[idx]
    print(energy_prob(events, WIMP, nucleus, norm_fact, E_r, del_Er))


if __name__ == "__main__":
    main()

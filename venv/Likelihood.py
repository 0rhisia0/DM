import functions as fn
import constants as const
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy import integrate

BULK = 1000 * 300


def events_likelihood(E_r, events, WIMP, A, E_thr, del_Er):
    p_energies = 1

    obs_events = len(events)  # num events observed
    E_r, int_rate = fn.integrate_rate(E_r, WIMP, A)
    idx = find_nearest_idx(E_r, E_thr)
    pred_events = int_rate[idx]*BULK
    poisson_prob = poisson.logpmf(obs_events, pred_events)

    e_prob = energy_prob(events, WIMP, A, E_r, del_Er)
    return poisson_prob+e_prob


def energy_prob(events, WIMP, A, E_r, del_Er):
    dif_rate = fn.diff_rate(E_r, WIMP, A)
    dif_rate /= np.sum(dif_rate)
    prob = 0
    for event in events:
        prob += np.log10(dif_rate[int(event)])
    return prob


def find_indices(E_r, events):
    new_events = np.zeros(len(events))
    for i in range(len(events)):
        idx = find_nearest_idx(E_r, events[i])
        new_events[i] = idx
    return new_events


def find_nearest_idx(array, value):
    array = array - value
    idx = (np.abs(array)).argmin()
    return idx


def main():
    Emin = 1 * const.keV
    Emax = fn.max_recoil_energy()
    Nsteps = 159
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
    nevents = find_indices(E_r, events)
    print(events_likelihood(E_r, events, WIMP, const.AXe, E_thr, del_Er))


if __name__ == "__main__":
    main()

import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def events_likelihood(e_min, e_max, n_steps, events, WIMP, A):
    WIMP = [("mass", "sigma")]
    del_Er = (e_max - e_min) / n_steps
    Er = np.arange(Emin, Emax, del_Er)
    p_energies = energy_prob(events, WIMP, A)


def energy_prob(events, WIMP, A):
    p_energies = 1
    dif_rate = fn.diff_rate2(WIMP, A)
    Er, int_rate = fn.integrate_rate(WIMP, A)
    p_d_f_energies = dif_rate / int_rate
    for event_energy in events:
        idx = find_nearest_idx(event_energy, Er)
        p_energies *= p_d_f_energies[idx]
    return p_energies


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def main():
    events = []
    events_likelihood()

if __name__=="__main__":
    main()
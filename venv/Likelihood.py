import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def events_likelihood(events, WIMP, A):
    WIMP = [("mass", "sigma")]
    expected_events = 
    for energy in events:
        p_energy = energy_prob(energy, WIMP, A)


def energy_prob(energy, WIMP, A):
    # Find the probability of each energy
    dif_rate = fn.diff_rate2(energy, WIMP, A)
    Er, int_rate = fn.integrate_rate(WIMP, A)
    return dif_rate/int_rate






def main():
    events = []
    events_likelihood()

if __name__=="__main__":
    main()
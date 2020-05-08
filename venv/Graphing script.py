import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')

def plot_diff_rate():
    Emin = 0.001 * const.keV
    Emax = 200. * const.keV
    Nsteps = 10000
    conv_fact = const.kg*const.day*const.keV/(const.c**2)
    del_Er = (Emax - Emin) / Nsteps
    Er = np.arange(Emin, Emax, del_Er)
    Xe = conv_fact*fn.diff_rate(Er, const.AXe)
    Ge = conv_fact*fn.diff_rate(Er, const.AGe)
    Ar = conv_fact*fn.diff_rate(Er, const.AAr)
    fig, ax = plt.subplots()
    line1, = ax.plot(Er, Xe, label='Xe')
    line2, = ax.plot(Er, Ge, label='Ge')
    line3, = ax.plot(Er, Ar, label='Ar')
    ax.legend()
    plt.xlabel("Recoil Energy (KeV)")
    plt.ylabel("Differential Rate (counts/kg/day/KeV)")
    plt.yscale('log')
    plt.show()


def plot_int_rate2():
    xXe, yXe = fn.integrate_rate_new(const.AXe)
    xAr, yAr = fn.integrate_rate_new(const.AAr)
    xGe, yGe = fn.integrate_rate_new(const.AGe)
    fig, ax = plt.subplots()
    line1, = ax.plot(xXe, yXe, label='Xe')
    line2, = ax.plot(xGe, yGe, label='Ge')
    line3, = ax.plot(xAr, yAr, label='Ar')
    ax.set_xlim(1.e-2, 200)
    ax.set_ylim(1.e-27, 1e-24)
    plt.legend()
    plt.ylabel("Rate (counts/kg/day)")
    plt.xlabel("Threshold Energy (KeV)")
    plt.yscale('log')
    plt.show()


def main():
    # Emax = 544**2*2*(const.Mn*const.AXe*const.M_D/(const.M_D+const.Mn*const.AXe))**2/(const.Mn*const.AXe)
    # print(Emax)
    # plot_diff_rate()
    plot_int_rate2()
    # plt.show()



if __name__=="__main__":
    main()


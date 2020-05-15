import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')

def plot_rate():
    Emin = 0.001 * const.keV
    Emax = 200. * const.keV
    Nsteps = 10000
    del_Er = (Emax - Emin) / Nsteps
    Er = np.arange(Emin, Emax, del_Er)
    Xe = fn.diff_rate(Er, const.AXe)
    Ge = fn.diff_rate(Er, const.AGe)
    Ar = fn.diff_rate(Er, const.AAr)
    fig, ax = plt.subplots()
    line1, = ax.plot(Er, Xe, label='Xe')
    line2, = ax.plot(Er, Ge, label='Ge')
    line3, = ax.plot(Er, Ar, label='Ar')
    xXe, yXe = fn.integrate_rate_new(const.AXe)
    xAr, yAr = fn.integrate_rate_new(const.AAr)
    xGe, yGe = fn.integrate_rate_new(const.AGe)
    line4, = ax.plot(xXe, yXe, label='Xe', linestyle='--')
    line5, = ax.plot(xGe, yGe, label='Ge', linestyle='--')
    line6, = ax.plot(xAr, yAr, label='Ar', linestyle='--')
    ax.set_xlim(0, 200)
    ax.set_ylim(1e-6, 1e-3)
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
    ax.set_xlim(1.e-2, 60)
    # ax.set_ylim(1.e-5, 1e-3)
    plt.legend()
    plt.ylabel("Rate (counts/kg/day)")
    plt.xlabel("Threshold Energy (KeV)")
    plt.yscale('log')
    plt.show()


def main():
    # plot_rate()
    plot_int_rate2()
    # plt.show()



if __name__=="__main__":
    main()


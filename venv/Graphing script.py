import functions as fn
import constants as const
import numpy as np
import scipy as sp
from scipy import integrate
from matplotlib import pyplot as plt

plt.style.use('ggplot')

def plot_int_rate2():
    nuclei_name = ["Xe", "Ar", "Ge"]
    nuclei = [const.AXe, const.AAr, const.AGe]
    WIMP = [const.M_D, const.sigma]
    E_min = 0.001*const.keV
    Nsteps = 1000
    E_max = []
    for nucleus in nuclei:
        E_max.append(fn.max_recoil_energy(nucleus))
        print("E_max for ", nucleus, "=", fn.max_recoil_energy(nucleus))
    fig, ax = plt.subplots()
    for i in range(len(nuclei)):
        del_Er = (E_max[i] - E_min) / Nsteps
        E_r = np.arange(E_min, E_max[i], del_Er)
        y = fn.diff_rate(E_r, WIMP, nuclei[i])
        ax.plot(E_r, y, label=nuclei_name[i])
    # for i in range(len(nuclei)):
    #     del_Er = (E_max[i] - E_min) / Nsteps
    #     E_r = np.arange(E_min, E_max[i], del_Er)
    #     x, y = fn.integrate_rate(E_r, WIMP, nuclei[i])
    #     ax.plot(x, y, label=nuclei_name[i], linestyle="--")
    # ax.set_xlim(1.e-2, 60)
    ax.set_ylim(1.e-8, 1e-2)
    ax.legend()
    plt.ylabel("Rate (counts/kg/day)")
    plt.xlabel("Threshold Energy (KeV)")
    plt.yscale('log')
    plt.show()


def main():
    plot_int_rate2()

if __name__=="__main__":
    main()


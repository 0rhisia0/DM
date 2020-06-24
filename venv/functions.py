import numpy as np
import scipy.special as sp
import constants as const
import math
import scipy.integrate as integrate
erf = sp.erf


# Performs the velocity integral for the WIMP distribution
def vel_integral(M_T, E_r, mu):
    # all primary issues are here
    v_min_arr = const.c * np.sqrt(M_T * E_r / (2 * mu ** 2))
    k = ((np.pi ** (3 / 2)) * (const.v_0 ** 3) * (
            erf(const.v_esc / const.v_0) - 2 * const.v_esc * np.exp(-(const.v_esc ** 2 / const.v_0 ** 2)) / (
            np.sqrt(math.pi) * const.v_0))) ** (-1)
    integral_v = np.zeros(v_min_arr.shape[0])
    for i in range(v_min_arr.shape[0]):
        g = 0
        v_min = v_min_arr[i]
        if v_min <= const.v_esc - const.v_e:
            g = erf((const.v_e - v_min) / const.v_0) + erf((const.v_e + v_min) / const.v_0) - 4 * const.v_e * np.exp(
                -const.v_esc ** 2 / (const.v_0 ** 2)) / (
                        np.sqrt(np.pi) * const.v_0)
        elif const.v_esc - const.v_e < v_min <= const.v_esc + const.v_e:
            g = erf((const.v_e - v_min) / const.v_0) + erf(const.v_esc / const.v_0) - 2 * (
                    const.v_e + const.v_esc - v_min) * np.exp(
                -const.v_esc ** 2 / (const.v_0 ** 2)) / (np.sqrt(np.pi) * const.v_0)
        integral_v[i] = g * (np.pi ** (3 / 2) * const.v_0 ** 3 * k) / (2 * const.v_e)
    return integral_v


# Helmholtz form factor implemented using trig functions
def form_factor(Er, A):
    M_T = const.Mn * A
    q = np.sqrt(2. * M_T * Er)
    r = np.sqrt((1.23*A**(1/3)-0.60)**2-2.18) * const.fm
    qr = q * r / const.hc
    qs = q * const.skin
    return 3. * (np.sin(qr) - qr * np.cos(qr)) / (qr ** 3.) * np.exp(-(qs ** 2) / 2.)


# determines E_max or the upper bound to the integration
def max_recoil_energy():
    M_D = const.M_D
    return 200
    # return M_D*const.v_esc**2/(2*const.c**2)


# calculates the integrated WIMP nuclear recoil rates
def integrate_rate(E_r, WIMP, A):
    dif_rate = diff_rate(E_r, WIMP, A)
    x = np.empty(E_r.shape[0])
    y = np.empty(E_r.shape[0])
    for i in range(E_r.shape[0]):
        domain = E_r[i:]
        codomain = dif_rate[i:]
        integral = integrate.simps(codomain, domain)
        x[i] = E_r[i]
        y[i] = integral
    return x, y


def diff_rate(E_r, WIMP, A):
    M_D = WIMP[0]
    sigma = WIMP[1]
    conv_fact = const.kg * const.day * const.keV * (const.c ** 2)
    M_T = const.Mn * A
    mu = const.Mn * A * M_D / (M_D + const.Mn * A)
    mun = const.Mn * M_D / (M_D + const.Mn)
    constant = const.N0 * sigma * const.rho * M_T / (2 * mun ** 2 * M_D)
    return constant * A * form_factor(E_r, A) ** 2 * vel_integral(M_T, E_r, mu) * conv_fact


# def velDist1(v_D):
#     k = ((math.pi ** (3 / 2)) * (const.v_0 ** 3) * (
#             erf(const.v_esc / const.v_0) - 2 * const.v_esc * math.exp(-(const.v_esc / const.v_0) ** 2) / (
#             math.sqrt(math.pi) * const.v_0))) ** (-1)
#     const = k * math.pi * const.v_0 ** 2 * v_D / const.v_e
#     func = math.exp(-(v_D - const.v_e) ** 2 / (const.v_0 ** 2)) - math.exp(-(v_D + const.v_e) ** 2 / (const.v_0 ** 2))
#     return const * func
#
#
# def velDist2(v_D):
#     k = ((math.pi ** (3 / 2)) * (const.v_0 ** 3) * (
#             erf(const.v_esc / const.v_0) - 2 * const.v_esc * math.exp(-(const.v_esc / const.v_0) ** 2) / (
#             math.sqrt(math.pi) * const.v_0))) ** (-1)
#     const = k * math.pi * const.v_0 ** 2 * v_D / const.v_e
#     func = math.exp(-(v_D - const.v_e) ** 2 / (const.v_0 ** 2)) - math.exp(-(const.v_esc) ** 2 / (const.v_0 ** 2))
#     return const * func
#
#
# def velDist3(v_D):
#     return 0


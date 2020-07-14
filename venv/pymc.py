import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import constants as const
import scipy.stats as stats
import nestpy

# This is same as C++ NEST with naming
nc = nestpy.NESTcalc(nestpy.VDetector())

interaction = nestpy.INTERACTION_TYPE(0)  # NR

E = 10  # keV
print('For an %s keV %s' % (E, interaction))

# Get particle yields
y = nc.GetYields(interaction,
		 E)

print('The photon yield is:', y.PhotonYield)
print('With statistical fluctuations',
      nc.GetQuanta(y).photons)
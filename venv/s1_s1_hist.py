import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import constants as const
import scipy.stats as stats

data = np.load('s1_s2_data.npy')
S1_min = np.min(data[:, 0])
S1_max = np.max(data[:, 0])
print(S1_max)
plt.hexbin(data[:, 0], np.log10(data[:, 1]))
plt.xlabel("S1")
plt.ylabel("log(S2)")
plt.show()

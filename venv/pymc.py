import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import constants as const
import scipy.stats as stats

data = np.load('s1_s2_data.npy')
plt.hexbin(data[:, 0], np.log(data[:, 1]))
plt.xlabel("S1")
plt.ylabel("log(S2)")
plt.show()

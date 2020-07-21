import nestpy
import numpy as np
import scipy as sp

with open('mock.txt') as f:
    events = f.read().splitlines()
events = np.asarray([float(num) for num in events])
print(events.shape)
detector = nestpy.DetectorExample_XENON10()
detector.Initialization()

# nc = nestpy.testNEST(detector, 10, 'NR', 100., 120., 10., "0., 0., 0.", "120.", -1., 1, True, 1.0)

# print(nc[0])
nc = nestpy.NESTcalc(detector)
interaction = nestpy.INTERACTION_TYPE(0)
yields = np.empty((events.shape[0], 2))
g1 = detector.get_g1()
g2 = nc.CalculateG2(False)
electron = np.random.choice(events, events.shape, replace=True)
for i in range(events.shape[0]):
    y = nc.GetYields(interaction, events[i])
    x = nc.GetQuanta(y)
    s1 = x.photons*g1
    s2 = x.electrons*g2[3]
    yields[i, 0] = s1
    yields[i, 1] = s2
np.save('s1_s2_data', yields)



from seispy.synthetics import synlv
from seispy.filters import bfilt
from seispy.plotting import wiggle
import matplotlib.pyplot as plt
import numpy as np


segy = synlv().to_memory()
wiggle(segy)
plt.show()

bandpassed = segy | bfilt()

im_dat = np.array([trace for trace in bandpassed])

plt.figure()
plt.imshow(im_dat.T, clim=[-1, 1], cmap='seismic')
plt.show()
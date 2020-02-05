import numpy as np
import os
import matplotlib.pyplot as py
from glob import glob

fnames = glob('trained_models/*.txt')

dice     = []
dice_std = []

for fname in fnames:
  data = np.loadtxt(fname, delimiter = ' ')
  dice.append(data[:,0])
  dice_std.append(data[:,1])

fig, ax = py.subplots(figsize = (6,3))

for i, fname in enumerate(fnames):
  ax.errorbar(np.arange(10) + 0.1*i, dice[i], yerr = dice_std[i],
              label=os.path.basename(fname)[16:-13], fmt = '.')

ax.set_ylim(0.5,1)
ax.legend(loc=3)
ax.grid()
ax.set_xlabel('class')
ax.set_ylabel('dice coefficient')
fig.tight_layout()
fig.show()



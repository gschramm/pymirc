# small demo on how to convert a voxelized segmentation
# to a 3D surface triangulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

import nibabel as nib

n     = 30
x,y,z = np.meshgrid(np.arange(n),np.arange(n),np.arange(n))
r     = np.sqrt((x-0.5*n)**2 + (y-0.5*n)**2 + (z-0.5*n)**2)

vol = np.zeros((n,n,n))
vol[r<0.3*n] = 1
vol[r<0.1*n] = 0

voxsize = np.array([1.,1.,1.])

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes_lewiner(vol, 0, spacing = voxsize)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

vmax = verts.max(axis=0)

ax.set_xlim(0, 1.2*vmax[0])
ax.set_ylim(0, 1.2*vmax[1])  
ax.set_zlim(0, 1.2*vmax[2])

plt.tight_layout()
plt.show()

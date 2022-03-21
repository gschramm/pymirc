import pylab as py
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.widgets      import Slider, TextBox

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class ThreeAxisViewer:
  """ simplistic three axis viewer for multiple aligned 3d or 4d arrays

  Parameters
  ----------
  vols : list
    a list of 3d or 4d numpy arrays containing the image volumes
  vxosize: list, optional
    a 3 element with the voxel size
  width: float, optional
    width of the figure
  sl_x, sl_y, sl_z, sl_t: int, optional
    slices to show at beginning
  ls : string, optional
     str specifying the line style of the cross hair (use '' for no cross hair)
  imshow_kwargs : list of dictionaries
     list of dictionaries with keyword arguments passed to pylab.imshow()
  rowlabels : list of strings
     containing the labels for every row (volume)

  Note
  ----
  Scrolling with the mouse or the up and down keys can be used to scroll thorugh the slices.
  The left and right keys scroll through time frames.
  The viewer expects the input volumes to be in LPS orientation.
  If the input is 4D, the time axis should be the left most axis.

  Example
  -------
  ims_kwargs = [{'vmin':-1,'vmax':1},{'vmin':-2,'vmax':2,'cmap':py.cm.jet}]
  vi = ThreeAxisViewer([np.random.randn(90,90,80),np.random.randn(90,90,80)], imshow_kwargs = ims_kwargs)
  """
  def __init__(self, vols, 
                     ovols          = None,
                     voxsize        = [1.,1.,1.], 
                     width          = None, 
                     sl_x           = None, 
                     sl_y           = None, 
                     sl_z           = None, 
                     sl_t           = 0, 
                     ls             = ':',
                     rowlabels      = None,
                     imshow_kwargs  = {},
                     oimshow_kwargs = {}):

    # image volumes
    if not isinstance(vols,list): 
      self.vols = [vols]
    else:
      self.vols = vols

    self.n_vols = len(self.vols)
    self.ndim   = self.vols[0].ndim

    # overlay volumes
    if ovols is None:
      self.ovols = None
    else:
      if not isinstance(ovols,list): 
        self.ovols = [ovols]
      else:
        self.ovols = ovols

    # factor used to hide / show overlays
    self.ofac = 1 

    if self.ndim: self.nframes = self.vols[0].shape[0]
    else:         self.nframes = 1 

    # set up the slice objects for correct slicing of 3d and 4d arrays
    if self.ndim == 3:
      self.ix = 0
      self.iy = 1
      self.iz = 2
    
    if self.ndim == 4:
      self.ix = 1
      self.iy = 2
      self.iz = 3
   
    self.shape = self.vols[0].shape

    if sl_x is None: 
      self.sl_x = self.shape[self.ix] // 2
    else:
      self.sl_x = sl_x

    if sl_y is None: 
      self.sl_y = self.shape[self.iy] // 2
    else:
      self.sl_y = sl_y

    if sl_z is None: 
      self.sl_z = self.shape[self.iz] // 2
    else:
      self.sl_z = sl_z

    self.sl_t = sl_t

    if self.ndim == 4:
      self.fstr = ', ' + str(self.sl_t)
    else:
      self.fstr = ''

    # kwargs for real volumes
    self.imshow_kwargs = imshow_kwargs

    if not isinstance(self.imshow_kwargs,list): 
      tmp = self.imshow_kwargs.copy()
      self.imshow_kwargs = []
      for i in range(self.n_vols): self.imshow_kwargs.append(tmp.copy())

    for i in range(self.n_vols):
      if not 'cmap' in self.imshow_kwargs[i]:          
        self.imshow_kwargs[i]['cmap'] = py.cm.Greys
      if not 'interpolation' in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['interpolation'] = 'nearest'
      if not 'vmin'          in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['vmin'] = self.vols[i].min()
      if not 'vmax'          in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['vmax'] = self.vols[i].max()
    
    # overlay imshow kwargs  
   
    self.oimshow_kwargs = oimshow_kwargs
    if not isinstance(self.oimshow_kwargs,list): 
      tmp = self.oimshow_kwargs.copy()
      self.oimshow_kwargs = []
      for i in range(self.n_vols): self.oimshow_kwargs.append(tmp.copy())

    for i in range(self.n_vols):
      if not 'cmap' in self.oimshow_kwargs[i]:          
        self.oimshow_kwargs[i]['cmap'] = py.cm.hot
      if not 'alpha' in self.oimshow_kwargs[i]:          
        self.oimshow_kwargs[i]['alpha'] = 0.5
        if not 'interpolation' in self.oimshow_kwargs[i]: 
          self.oimshow_kwargs[i]['interpolation'] = 'nearest'
      if self.ovols is not None:
        if self.ovols[i] is not None:
          if not 'vmin'          in self.oimshow_kwargs[i]: 
            self.oimshow_kwargs[i]['vmin'] = self.ovols[i].min()
          if not 'vmax'          in self.oimshow_kwargs[i]: 
            self.oimshow_kwargs[i]['vmax'] = self.ovols[i].max()

    # generat the slice objects sl0, sl2, sl2
    self.recalculate_slices()
   
    # set up the figure with the images
    if width == None: width = min(12,24/len(self.vols))
    fig_asp = self.n_vols*max(self.shape[self.iy], 
               self.shape[self.iz]) / (2*self.shape[self.ix] + self.shape[self.iy])

    self.fig, self.ax = py.subplots(self.n_vols, 3, figsize = (width,width*fig_asp), squeeze = False)
    self.axes         = self.fig.get_axes()

    self.imgs  = []
    self.oimgs = None

    for i in range(self.n_vols):
      im0 = np.squeeze(self.vols[i][tuple(self.sl0)].T)
      im1 = np.squeeze(np.flip(self.vols[i][tuple(self.sl1)].T,0))
      im2 = np.squeeze(np.flip(self.vols[i][tuple(self.sl2)].T,0))
 
      tmp  = []
      tmp.append(self.ax[i,0].imshow(im0, aspect=voxsize[1]/voxsize[0], **self.imshow_kwargs[i]))
      tmp.append(self.ax[i,1].imshow(im1, aspect=voxsize[2]/voxsize[0], **self.imshow_kwargs[i]))
      tmp.append(self.ax[i,2].imshow(im2, aspect=voxsize[2]/voxsize[1], **self.imshow_kwargs[i]))

      self.imgs.append(tmp)

      self.ax[i,0].set_axis_off()
      self.ax[i,1].set_axis_off()    
      self.ax[i,2].set_axis_off()
 

    if rowlabels is None:
      self.fig.subplots_adjust(left=0,right=0.95,bottom=0,top=0.97,wspace=0.01,hspace=0.01)

    else:
      for ivol, label in enumerate(rowlabels):
        self.fig.text(0.01,1 - (ivol + 0.5)/self.n_vols, label, rotation='vertical', 
                      size = 'large', verticalalignment = 'center')
      self.fig.subplots_adjust(left=0.03,right=0.95,bottom=0,top=0.97,wspace=0.01,hspace=0.01)
    self.cb_ax = []

    # align all subplots in a row to bottom and add axes for colorbars
    for irow in range(self.n_vols):
      bboxes = []
      for icol in range(3):
        bboxes.append(self.ax[irow,icol].get_position())
      y0s = [x.y0 for x in bboxes]  
      y0min = min(y0s)
 
      for icol in range(3):
        bbox = bboxes[icol]
        self.ax[irow,icol].set_position([bbox.x0, y0min, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

      self.cb_ax.append(inset_axes(self.ax[irow,-1], width=0.01*width, 
                        height=0.8*width*fig_asp/self.n_vols,
                        loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1),
                        bbox_transform=self.ax[irow,-1].transAxes, borderpad=0))


    # add the overlay images in case given
    if self.ovols is not None:
      self.oimgs = []
      for i in range(self.n_vols):
        if self.ovols[i] is not None:
          oim0    = np.squeeze(self.ovols[i][tuple(self.sl0)].T)
          oim1    = np.squeeze(np.flip(self.ovols[i][tuple(self.sl1)].T,0))
          oim2    = np.squeeze(np.flip(self.ovols[i][tuple(self.sl2)].T,0))
 
          tmp  = []
          tmp.append(self.ax[i,0].imshow(oim0, aspect=voxsize[1]/voxsize[0], **self.oimshow_kwargs[i]))
          tmp.append(self.ax[i,1].imshow(oim1, aspect=voxsize[2]/voxsize[0], **self.oimshow_kwargs[i]))
          tmp.append(self.ax[i,2].imshow(oim2, aspect=voxsize[2]/voxsize[1], **self.oimshow_kwargs[i]))

          self.oimgs.append(tmp)
        else:
          self.oimgs.append(None)

 
    self.ax[0,0].set_title(str(self.sl_z) + self.fstr, fontsize='small') 
    self.ax[0,1].set_title(str(self.sl_y) + self.fstr, fontsize='small') 
    self.ax[0,2].set_title(str(self.sl_x) + self.fstr, fontsize='small') 

    # add colors bars
    self.cb_top_labels = []
    self.cb_bottom_labels = []

    for i in range(self.n_vols): 
      self.cb_ax[i].imshow(np.arange(128).reshape((128,1)), aspect = 0.2, origin = 'lower',
                           cmap = self.imshow_kwargs[i]['cmap']) 
      self.cb_top_labels.append(self.cb_ax[i].text(1.2, 1, f'{self.imshow_kwargs[i]["vmax"]:.1E}', 
                         transform = self.cb_ax[i].transAxes, rotation = 90,
                         horizontalalignment = 'left', verticalalignment = 'top', size = 'small'))
      self.cb_bottom_labels.append(self.cb_ax[i].text(1.2, 0, f'{self.imshow_kwargs[i]["vmin"]:.1E}', 
                         transform = self.cb_ax[i].transAxes, rotation = 90,
                         horizontalalignment = 'left', verticalalignment = 'bottom', size = 'small'))
      self.cb_ax[i].set_xticks([])
      self.cb_ax[i].set_yticks([])


    # connect the image figure with actions
    self.fig.canvas.mpl_connect('scroll_event',self.onscroll)
    self.fig.canvas.mpl_connect('button_press_event',self.onbuttonpress)
    self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
    self.fig.show()

    # add cross hair
    self.l0x = [] 
    self.l0y = []
    self.l1x = []
    self.l1y = []
    self.l2x = []
    self.l2y = []
   
    self.showCross = True
    if ls == '': self.showCross = False

    if self.showCross:
      for i in range(self.n_vols):
        self.l0x.append(self.axes[3*i + 0].axvline(self.sl_x, color = 'r',ls = ls)) 
        self.l0y.append(self.axes[3*i + 0].axhline(self.sl_y, color = 'r',ls = ls))

        self.l1x.append(self.axes[3*i + 1].axvline(self.sl_x, color = 'r',ls = ls))
        self.l1y.append(self.axes[3*i + 1].axhline(self.shape[self.iz] - self.sl_z, color = 'r',ls = ls))

        self.l2x.append(self.axes[3*i + 2].axvline(self.sl_y, color = 'r',ls = ls))
        self.l2y.append(self.axes[3*i + 2].axhline(self.shape[self.iz] - self.sl_z, color = 'r',ls = ls))

    # list for contour definitions
    self.contour_configs = []

  #------------------------------------------------------------------------
  def update_colorbars(self):
    for i in range(self.n_vols): 
      self.cb_top_labels[i].set_text(f'{self.imshow_kwargs[i]["vmax"]:.1E}')
      self.cb_bottom_labels[i].set_text(f'{self.imshow_kwargs[i]["vmin"]:.1E}')

  #------------------------------------------------------------------------
  def set_vmin(self, i, val):
    if i < self.n_vols:
      self.imshow_kwargs[i]['vmin'] = val
      self.imgs[i][0].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][1].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][2].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])

      self.update_colorbars()
      self.fig.canvas.draw()

  #------------------------------------------------------------------------
  def set_vmax(self, i, val):
    if i < self.n_vols:
      self.imshow_kwargs[i]['vmax'] = val
      self.imgs[i][0].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][1].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][2].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])

      self.update_colorbars()
      self.fig.canvas.draw()

  #------------------------------------------------------------------------
  def redraw_transversal(self):
    for i in range(self.n_vols):
      self.imgs[i][0].set_data(np.squeeze(self.vols[i][tuple(self.sl0)].T))
      if (self.oimgs is not None) and (self.oimgs[i] is not None):
        self.oimgs[i][0].set_data(self.ofac*np.squeeze(self.ovols[i][tuple(self.sl0)].T))

    self.ax[0,0].set_title(str(self.sl_z) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l0x: l.set_xdata(self.sl_x) 
        for l in self.l0y: l.set_ydata(self.sl_y) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw_coronal(self):
    
    for i in range(self.n_vols):
      self.imgs[i][1].set_data(np.squeeze(np.flip(self.vols[i][tuple(self.sl1)].T,0)))
      if (self.oimgs is not None) and (self.oimgs[i] is not None):
        self.oimgs[i][1].set_data(self.ofac*np.squeeze(np.flip(self.ovols[i][tuple(self.sl1)].T,0)))
    self.ax[0,1].set_title(str(self.sl_y) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l1x: l.set_xdata(self.sl_x) 
        for l in self.l1y: l.set_ydata(self.shape[self.iz] - self.sl_z - 1) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw_sagittal(self):
    for i in range(self.n_vols):
      self.imgs[i][2].set_data(np.squeeze(np.flip(self.vols[i][tuple(self.sl2)].T,0)))
      if (self.oimgs is not None) and (self.oimgs[i] is not None):
        self.oimgs[i][2].set_data(self.ofac*np.squeeze(np.flip(self.ovols[i][tuple(self.sl2)].T,0)))

    self.ax[0,2].set_title(str(self.sl_x) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l2x: l.set_xdata(self.sl_y) 
        for l in self.l2y: l.set_ydata(self.shape[self.iz] - self.sl_z - 1) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw(self):
    self.redraw_transversal()
    self.redraw_coronal()
    self.redraw_sagittal()

    # draw all contour lines
    if len(self.contour_configs) > 0:
      for cfg in self.contour_configs:
        for i in range(3):
           # remove drawn contour lines first
          while(len(self.ax[cfg[1],i].collections) > 0):
            for col in self.ax[cfg[1],i].collections:
              col.remove()

          self.ax[cfg[1],i].contour(self.imgs[cfg[0]][i].get_array(), cfg[2], **cfg[3])
      self.fig.canvas.draw()
  #------------------------------------------------------------------------
  def recalculate_slices(self):
    if self.ndim == 3:
      self.sl0          = [slice(None)]*self.ndim
      self.sl0[self.iz] = slice(self.sl_z, self.sl_z+1)
      self.sl1          = [slice(None)]*self.ndim
      self.sl1[self.iy] = slice(self.sl_y, self.sl_y+1)
      self.sl2          = [slice(None)]*self.ndim
      self.sl2[self.ix] = slice(self.sl_x, self.sl_x+1)
    elif self.ndim == 4:
      self.sl0          = [slice(None)]*self.ndim
      self.sl0[self.iz] = slice(self.sl_z, self.sl_z+1)
      self.sl0[0]       = slice(self.sl_t, self.sl_t+1)
      self.sl1          = [slice(None)]*self.ndim
      self.sl1[self.iy] = slice(self.sl_y, self.sl_y+1)
      self.sl1[0]       = slice(self.sl_t, self.sl_t+1)
      self.sl2          = [slice(None)]*self.ndim
      self.sl2[self.ix] = slice(self.sl_x, self.sl_x+1)
      self.sl2[0]       = slice(self.sl_t, self.sl_t+1)
      self.fstr         = ', ' + str(self.sl_t)

  #------------------------------------------------------------------------
  def add_contour(self, source, target, levels, contour_kwargs):
    self.contour_configs.append([source, target, levels, contour_kwargs])
    self.redraw()

  #------------------------------------------------------------------------
  def remove_contour(self, k):
    if k < len(self.contour_configs):
      cfg = self.contour_configs[k]
      for i in range(3):
        # remove drawn contour lines first
        while(len(self.ax[cfg[1],i].collections) > 0):
          for col in self.ax[cfg[1],i].collections:
            col.remove()
      self.contour_configs.pop(k)
      self.redraw()

  #------------------------------------------------------------------------
  def onkeypress(self,event):
    if ((event.key == 'left' or event.key == 'right' or event.key == 'up' or event.key == 'down') and 
        (self.ndim >= 3)):
      if event.key == 'left' and self.ndim == 4:    
        self.sl_t = (self.sl_t - 1) % self.nframes 
        self.recalculate_slices()
        self.redraw()
      elif event.key == 'right' and self.ndim == 4: 
        self.sl_t = (self.sl_t + 1) % self.nframes 
        self.recalculate_slices()
        self.redraw()
      else:
        if event.inaxes in self.axes:
          iax = self.axes.index(event.inaxes)
          
          if (iax %3 == 0):
            if event.key == 'up':
                self.sl_z = (self.sl_z + 1) % self.shape[self.iz]
            elif event.key == 'down':
                self.sl_z = (self.sl_z - 1) % self.shape[self.iz]

            self.recalculate_slices()
            self.redraw()

          elif (iax %3 == 1):
            if event.key == 'up':
                self.sl_y = (self.sl_y + 1) % self.shape[self.iy]
            elif event.key == 'down':
                self.sl_y = (self.sl_y - 1) % self.shape[self.iy]

            self.recalculate_slices()
            self.redraw()

          elif (iax %3 == 2):
            if event.key == 'up':
                self.sl_x = (self.sl_x + 1) % self.shape[self.ix]
            elif event.key == 'down':
                self.sl_x = (self.sl_x - 1) % self.shape[self.ix]

            self.recalculate_slices()
            self.redraw()
    elif event.key == 'a':
      self.ofac = 1 - self.ofac 
      self.redraw()

  #------------------------------------------------------------------------
  def onbuttonpress(self,event):
    if py.get_current_fig_manager().toolbar.mode == '':
      if event.inaxes in self.axes:
        iax =  self.axes.index(event.inaxes)
        if iax < 3*self.n_vols:
          if iax % 3 == 0:
            self.sl_x = int(event.xdata) % self.shape[self.ix]
            self.sl_y = int(event.ydata) % self.shape[self.iy]
            self.recalculate_slices()
            self.redraw()
          elif iax % 3 == 1:
            self.sl_x = int(event.xdata) % self.shape[self.ix]
            self.sl_z = (self.shape[self.iz] - int(event.ydata)) % self.shape[self.iz]
            self.recalculate_slices()
            self.redraw()
          elif iax % 3 == 2:
            self.sl_y = int(event.xdata) % self.shape[self.iy]
            self.sl_z = (self.shape[self.iz] - int(event.ydata)) % self.shape[self.iz]
            self.recalculate_slices()
            self.redraw()

  #------------------------------------------------------------------------
  def onscroll(self,event):
    if event.inaxes in self.axes:
      iax = self.axes.index(event.inaxes)

      if (iax %3 == 0):
        if event.button == 'up':
            self.sl_z = (self.sl_z + 1) % self.shape[self.iz]
        elif event.button == 'down':
            self.sl_z = (self.sl_z - 1) % self.shape[self.iz]

        self.recalculate_slices()
        self.redraw()

      elif (iax %3 == 1):
        if event.button == 'up':
            self.sl_y = (self.sl_y + 1) % self.shape[self.iy]
        elif event.button == 'down':
            self.sl_y = (self.sl_y - 1) % self.shape[self.iy]

        self.recalculate_slices()
        self.redraw()

      elif (iax %3 == 2):
        if event.button == 'up':
            self.sl_x = (self.sl_x + 1) % self.shape[self.ix]
        elif event.button == 'down':
            self.sl_x = (self.sl_x - 1) % self.shape[self.ix]

        self.recalculate_slices()
        self.redraw()

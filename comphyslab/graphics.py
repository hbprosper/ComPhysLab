import os, sys
import numpy as np
import vpython as vp
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from types import SimpleNamespace as NS
from time import sleep
from comphyslab.utils import CircularBuffer
# ---------------------------------------------------------------------
# update fonts
FONTSIZE = 12
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# use latex if available on system, otherwise set usetex=False
import shutil
mp.rc('text', usetex=shutil.which('latex') is not None)

# use JavaScript for rendering animations
mp.rc('animation', html='jshtml')
# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
# in vpython, RGB colors must be defined using the vpython vector class
SKYBLUE   = vp.vector(0.62,0.57,0.98)
LAWNGREEN = vp.vector(0.50,0.90,0.50)
GRAY      = vp.vector(0.70,0.70,0.70)

WIDTH  = 500 # viewport width in pixels
HEIGHT = 300 # viewport height in pixels

ORIGIN = vp.vector(0,0,0)
I      = vp.vector(1,0,0) # unit vector in x direction
J      = vp.vector(0,1,0) # unit vector in y direction
K      = vp.vector(0,0,1) # unit vector in z direction
# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------
def create_canvas(caption, size, up=J):
    scene = vp.canvas()
    scene.caption = caption
    scene.width = WIDTH
    scene.height= HEIGHT
    scene.align = 'left'
    scene.background=SKYBLUE
    scene.userzoom = False            # disable vpython zoom; do our own
    scene.up = up                     # direction of vertical

    if up == J:
        r_camera = 2 * size * vp.vector(3,5,10).norm()
    else:
        r_camera = 2 * size * vp.vector(10,3,5).norm()
    scene.camera.axis =-r_camera
    scene.camera.pos  = r_camera
    scene.camera.dist = r_camera.mag
    return scene
    
# Implement zoom explicitly
def create_zoom(scene):
    def zoom_callback(s):
        distance = scene.camera.dist / s.value
        scene.camera.pos = distance * vp.norm(scene.camera.pos-scene.center)

    dist = scene.camera.dist
    scene.append_to_title('\nZoom ')
    zoom = vp.slider(min=0.5, max=2, value=1.0, 
                     pos=scene.title_anchor,
                           length=0.7*scene.width, bind=zoom_callback)
    return zoom

def draw_coordinate_system(size, draw_plane=True, up=J):
    axes = NS()
    sw = size/80
    aw = size/2

    # draw Cartesian axes 
    axes.xaxis = vp.arrow(pos=ORIGIN, axis=size*I, shaftwidth=sw, color=GRAY)
    axes.xlabel= vp.label(pos=aw*I, text='x', box=False) 
    
    axes.yaxis = vp.arrow(pos=ORIGIN, axis=size*J, shaftwidth=sw, color=GRAY)
    axes.ylabel= vp.label(pos=aw*J, text='y', box=False) 
    
    axes.zaxis = vp.arrow(pos=ORIGIN, axis=size*K, shaftwidth=sw, color=GRAY)
    axes.zlabel= vp.label(pos=aw*K, text='z', box=False) 
        
    if draw_plane:
        if up == J:
            n = I
            m = K
        else:
            n = J
            m = I
        a = vp.vertex(pos= size*n+size*m, color=LAWNGREEN, opacity=0.5)
        b = vp.vertex(pos= size*n-size*m, color=LAWNGREEN, opacity=0.5)
        c = vp.vertex(pos=-size*n-size*m, color=LAWNGREEN, opacity=0.5)
        d = vp.vertex(pos=-size*n+size*m, color=LAWNGREEN, opacity=0.5)
        axes.ground = vp.quad(vs=[a, b, c, d])
        
    return axes

def plot_central_xy_axes(ax, 
                         xmin, xmax, nxticks, xlabel,
                         ymin, ymax, nyticks, ylabel, 
                         ftsize=16):
    
    # location of xtick marks 
    xticks = np.linspace(xmin, xmax, nxticks)
    
    # get rid of the central tick mark count is odd
    if nxticks % 2 == 1:
        xticks = np.delete(xticks, nxticks//2)
        
    # location of ytick marks
    yticks = np.linspace(ymin, ymax, nyticks)
    if nyticks % 2 == 1:
        yticks = np.delete(yticks, nyticks//2)
        
    # define graph domain, tick marks, and labels
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks=xticks)
    ax.set_xlabel(xlabel, loc='right', fontsize=ftsize)

    # define graph range, tick marks, and labels  
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks=yticks)
    ax.set_ylabel(ylabel, loc='top', fontsize=ftsize)

    # move left y-axis and bottom x-axis to center of plot
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

# ---------------------------------------------------------------------
# CLASSES
# ---------------------------------------------------------------------
# Simple class to store state and widgets
class Bag(NS):
    def __init__(self, verbose=False):
        super().__init__()
        self.gfx = NS() # widgets stored in this "bag"
        self.verbose = verbose
        
    def clear(self):
        # free all graphics objects
        for key, obj in self.gfx.__dict__.items():
            try:
                obj.delete()
                if self.verbose:
                    print(f'\tdeleted: {key}')
            except:
                pass
        # clear all references to graphics objects.
        self.gfx.__dict__.clear()
        
class Sim:
    def __init__(self, context, update, 
                 stopped_message='Animation ended!', 
                 wait_before_delete=5):

        # verify context
        try:
            y = context.active
        except:
            context.active = True
            print('''
    Attribute "active" missing from "bag"! It has been added
    and set to True. 
            ''')
            
        try:
            y = context.update
        except:
            context.update = True
            print('''
    Attribute "update" missing from bag! It has been added and 
    set to True.
            ''')

        try:
            y = context.rate
        except:
            context.rate = 50
            print('''
    Attribute "rate" missing from bag! It has been added and
    set so that the frame rate is no greater than 50 frames / second.
            ''')
            
        self.context = context
        self.update  = update
        self.message = stopped_message
        self.wait_before_delete = wait_before_delete
        
    def run(self):
        
        while self.context.active:
            
            if self.context.update:
                self.update(self.context)
                
            vp.rate(self.context.rate)

        print(self.message)
        sleep(self.wait_before_delete)

        # free all graphics objects
        self.context.clear()
        
class Controls:
    def __init__(self, context):
        self.context = context
        
    def start_pause(self, b):
        self.context.update = not self.context.update
        if self.context.update:
            b.text='Pause'
            b.background=vp.color.white
        else:
            b.text='Start'
            b.background=vp.color.green

    def stop(self, b):
        self.context.active = False
        b.text='Stop'
        b.background=vp.color.white

    def recenter(self, b):
        scene = self.context.gfx.scene
        r_camera = scene.camera.dist * CAMERA
        scene.camera.axis =-r_camera
        scene.camera.pos  = r_camera

class Histogram:
    def __init__(self, title, xtitle, ytitle, xmin, xmax, 
                 nbins=50, 
                 color=vp.color.blue, 
                 density=True,
                 buffer_size=-1,
                 width=400, 
                 height=HEIGHT, 
                 align='right', fast=True):

        self.xmin    = xmin
        self.xmax    = xmax
        self.nbins   = nbins
        self.density = density
        self.delta   = (xmax-xmin)/nbins
        
        self.edges   = np.linspace(xmin, xmax, nbins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:])/2
        self.counts  = np.zeros(nbins, dtype=int)
        
        self.g = vp.graph(
            title=title, xtitle=xtitle, ytitle=ytitle,
            xmin=xmin, xmax=xmax, 
            align=align, width=width, height=height,
            fast=fast) # draw simpler graph for speed
        
        # bar width ~ bin width
        self.bars = vp.gvbars(graph=self.g, 
                              delta=self.delta, 
                              color=color)

        if buffer_size > 0:
            self.buffer = CircularBuffer(buffer_size, nbins)
        else:
            self.buffer = None
        
    def __del__(self):
        self.bars.delete()
        self.g.delete()

    def delete(self):
        self.__del__()
        
    def fill(self, x):
        # Find bin index
        II = np.floor((x-self.xmin) / self.delta).astype(np.int32)
        II = II[(0 <= II) * (II < self.nbins)]

        # Update counts
        newcounts = np.bincount(II, minlength=len(self.counts))
        
        if type(self.buffer) != type(None):
            self.buffer.append(newcounts)
            oldest_counts = self.buffer.get_oldest()
            if type(oldest_counts) != type(None):
                self.counts[:] = self.counts - oldest_counts
                
        self.counts[:] = self.counts + newcounts

    def clear(self):
        self.counts.fill(0)
        
    def draw(self):
        # update bar data
        if self.density:
            A = 1.0/self.counts.sum()
        else:
            A = 1.0
        self.bars.data = list(zip(self.centers, A*self.counts))

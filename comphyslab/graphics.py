import os, sys
import numpy as np
import vpython as vp
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
# in vpython, RGB colors must be defined using the vpython vector class
SKYBLUE   = vp.vector(0.62,0.57,0.98)
LAWNGREEN = vp.vector(0.50,0.90,0.50)
GRAY      = vp.vector(0.70,0.70,0.70)

WIDTH  = 600 # viewport width in pixels
HEIGHT = 300 # viewport height in pixels

ORIGIN = vp.vector(0,0,0)
I      = vp.vector(1,0,0) # unit vector in x direction
J      = vp.vector(0,1,0) # unit vector in y direction
K      = vp.vector(0,0,1) # unit vector in z direction
CAMERA = vp.vector(-0.1, -0.5, -0.8).norm() # direction in which camera points
# ---------------------------------------------------------------------
def create_canvas(caption, size):
    scene = vp.canvas()
    scene.caption = caption
    scene.range = size      # window size in world coordinates
    scene.width = WIDTH
    scene.height= HEIGHT
    scene.background=SKYBLUE
    scene.userzoom = False  # user can't zoom
    scene.up = J             # direction of vertical
    scene.forward= CAMERA    # direction in which camera looks
    return scene
    
def draw_coordinate_system(size):
    class Axes:
        pass
    axes = Axes()

    sw = size/80
    aw = size/2
    
    # draw ground
    a = vp.vertex( pos= size*I+size*K, color=LAWNGREEN, opacity=0.5)
    b = vp.vertex( pos= size*I-size*K, color=LAWNGREEN, opacity=0.5)
    c = vp.vertex( pos=-size*I-size*K, color=LAWNGREEN, opacity=0.5)
    d = vp.vertex( pos=-size*I+size*K, color=LAWNGREEN, opacity=0.5)
    axes.xzplane = vp.quad(vs=[a, b, c, d])
    
    # draw Cartesian axes 
    axes.xaxis = vp.arrow(pos=ORIGIN, axis=size*I, shaftwidth=sw, color=GRAY)
    axes.xlabel= vp.label(pos=aw*I, text='x', box=False) 
    
    axes.yaxis = vp.arrow(pos=ORIGIN, axis=size*J, shaftwidth=sw, color=GRAY)
    axes.ylabel= vp.label(pos=aw*J, text='y', box=False) 
    
    axes.zaxis = vp.arrow(pos=ORIGIN, axis=size*K, shaftwidth=sw, color=GRAY)
    axes.zlabel= vp.label(pos=aw*K, text='z', box=False) 

    return axes

class Controls:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def start_pause(self, b):
        self.cfg.update = not self.cfg.update
        if self.cfg.update:
            b.text='Pause'
            b.background=vp.color.white
        else:
            b.text='Start'
            b.background=vp.color.green

    def stop(self, b):
        self.cfg.active =False

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

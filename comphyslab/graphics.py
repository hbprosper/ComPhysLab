import os, sys
import numpy as np
import vpython as vp
import matplotlib as mp
import matplotlib.pyplot as plt
import subprocess

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
def Scene(title, size,
                  background=SKYBLUE,
                  up=J, 
                  height=HEIGHT, 
                  width=-1, 
                  camera=vp.vector(10,3,5)):
    scene = vp.canvas()
    scene.title = title
    scene.height= height
    if width > 0:
        scene.width = width
    else:
        scene.width = int(height * 16 / 9)
    scene.align = 'left'
    scene.background=background
    scene.userzoom = False            # disable vpython zoom; do our own
    scene.up = up                     # direction of vertical

    if up == J:
        x, y, z = camera.x, camera.y, camera.z
        camera = vp.vector(y,z,x)
        
    r_camera = 2 * size * camera.norm()
    scene.camera.axis =-r_camera
    scene.camera.pos  = r_camera
    scene.camera.dist = r_camera.mag
    return scene
create_canvas = Scene

# Implement zoom explicitly
def Zoom(scene):
    def zoom_callback(s):
        distance = scene.camera.dist / s.value
        scene.camera.pos = distance * vp.norm(scene.camera.pos-scene.center)

    dist = scene.camera.dist
    scene.append_to_title('\nZoom ')
    zoom = vp.slider(min=0.5, max=2, value=1.0, 
                     pos=scene.title_anchor,
                           length=0.7*scene.width, bind=zoom_callback)
    return zoom
create_zoom = Zoom

def CoordinateSystem(size, draw_plane=True, up=J):
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
draw_coordinate_system = CoordinateSystem

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
# ----------------------------------------------------------------------    
def plot_planet_positions(x, y, colors, text, 
                          filename='planets.png', 
                          title='Inner Solar System',
                          xmin=-2, xmax=2, ymin=-2, ymax=2, 
                          fgsize=(5,5), ftsize=16):
    
    # Set size of figure
    fig = plt.figure(figsize=fgsize)

    # Create area for a single plot 
    nrows, ncols, index = 1, 1, 1
    ax  = plt.subplot(nrows, ncols, index)

    ax.set_title(title, pad=14)
    
    # Place axes at the center of the plot
    nxticks = nyticks = 9
    xlabel  = '$x$ (au)'
    ylabel  = '$y$ (au)'
    plot_central_xy_axes(ax, 
                      xmin, xmax, nxticks, xlabel,
                      ymin, ymax, nyticks, ylabel, 
                      ftsize)
    
    ax.scatter(x, y, s=50, c=colors)
    
    for X, Y, T in zip(x, y, text):
        X += 0.1
        Y += 0.1
        ax.text(X, Y, T)
    
    fig.tight_layout()
    
    plt.savefig(filename)
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

def start_movie_export(scene, height=200, folder="frames"):
    """
    Prepare scene and directory for movie export.
    """
    scene.height = height
    scene.width = int(height * 16 / 9)
    scene.autoscale = False

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def capture_frame(scene, folder, frame_number):
    """
    Save a frame as PNG with zero-padded numbering.
    """
    filename = os.path.join(folder, f"frame_{frame_number:05d}.png")
    scene.capture(filename)
    
class Sim:
    def __init__(self, context, update, 
                 stopped_message='Animation ended!',
                 save_frames=False,
                 max_frames=-1,
                 wait_before_delete=3):
           
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
            context.rate = 30
            print('''
    Attribute "rate" missing from bag! It has been added and
    set so that the frame rate is no greater than 50 frames / second.
            ''')
 
        self.context = context
        self.update  = update
        self.message = stopped_message
        
        self.save_frames = save_frames
        self.context.max_frames = max_frames
        self.wait_before_delete = wait_before_delete

        if self.save_frames:
            if "ipykernel" in sys.modules:
                print("Running in Jupyter — scene.capture() disabled")
                self.save_frames = False
            else:
                self.folder = start_movie_export(self.context.gfx.scene)
                if context.max_frames < 0:
                    context.max_frames = 10 * context.rate
                    print('''
    You have called Sim with save_frames to True, but max_frames is not set. 
    max_frames has been set to 10 * rate. 
            ''')
 
    def run(self):
        self.context.frame = 0
        
        while self.context.active:

            vp.rate(self.context.rate)
            
            if self.context.update:
                
                self.update(self.context)

                if self.save_frames:
                    capture_frame(self.context.gfx.scene, self.folder, self.context.frame)
                    
                    if self.context.max_frames > 0:
                        self.context.active = self.context.frame >= self.context.max_frames-1

                self.context.frame += 1
                
        print(self.message)
        sleep(self.wait_before_delete)

        # free all graphics objects
        self.context.clear()

    def save(self, ratescale=1, filename='simulation'):
        if "ipykernel" in sys.modules:
            print('''
    Running in Jupyter — no frames have been saved! If you want to
    save frames to create an mp4 movie, save this notebook as a
    Python script and run the script using the python command:
    
        python name-of-saved-script

            ''')
            return
                        
        framerate = ratescale * self.context.rate
        encoders  = subprocess.check_output(["ffmpeg", "-encoders"], text=True)

        if "h264_videotoolbox" in encoders:
            codec = "h264_videotoolbox"
        elif "libx264" in encoders:
            codec = "libx264"
        else:
            raise RuntimeError("No suitable H.264 encoder found.")
        
        cmd = [
            "ffmpeg",
            "-framerate", f"{framerate}",
            "-i", "frames/frame_%05d.png",
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            f"{filename}.mp4"
        ]
        print(f'\nsaving as {filename}.mp4')
        subprocess.run(cmd)
        print('\ndone!')

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
    def __init__(self, title, xtitle, ytitle, nbins, xmin, xmax, 
                 ymin=None, ymax=None,
                 color=vp.color.blue, 
                 bufsize=-1,
                 width=250, 
                 height=200,
                 align='right', 
                 density=False,
                 fast=True):

        self.nbins   = nbins
        self.xmin    = xmin
        self.xmax    = xmax
        self.ymin    = ymin
        self.ymax    = ymax

        self.delta   = (xmax-xmin)/nbins
        self.density = density
        
        self.edges   = np.linspace(xmin, xmax, nbins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:])/2
        self.counts  = np.zeros(nbins, dtype=float)

        if ymin == None:
            self.g = vp.graph(
                title=title, xtitle=xtitle, ytitle=ytitle,
                xmin=xmin, xmax=xmax, 
                align=align, width=width, height=height,
                fast=fast) # draw simpler graph for speed
        else:
            self.g = vp.graph(
                title=title, xtitle=xtitle, ytitle=ytitle,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                align=align, width=width, height=height,
                fast=fast) # draw simpler graph for speed
            
        # bar width ~ bin width
        self.bars = vp.gvbars(graph=self.g, 
                              delta=self.delta, 
                              color=color)

        if bufsize > 0:
            self.buffer = CircularBuffer(bufsize, nbins)
            self.has_buffer = True
        else:
            self.buffer = None
            self.has_buffer = False
        
    def __del__(self):
        self.bars.delete()
        self.g.delete()

    def delete(self):
        self.__del__()
        
    def fill(self, x, binned=False):
        if binned:
            newcounts = x
        else:
            # Find bin index
            II = np.floor((x-self.xmin) / self.delta).astype(np.int32)
            II = II[(0 <= II) * (II < self.nbins)]
            newcounts = np.bincount(II, minlength=len(self.counts))
            
        if self.has_buffer:
            self.buffer.append(newcounts)
            oldest_counts = self.buffer.get_oldest()
            if type(oldest_counts) != type(None):
                self.counts[:] = np.maximum(self.counts - oldest_counts, 0)
            self.counts += newcounts
        else:      
            self.counts[:] = newcounts

    def clear(self):
        self.counts.fill(0)
        
    def draw(self):
        A = 1.0
        if self.has_buffer:
            A = 1.0 / self.buffer.count
            
        if self.density:
            A = 1.0 / self.counts.sum()

        self.bars.data = list(zip(self.centers, A*self.counts))
        
class Graph:
    def __init__(self, title, xtitle, ytitle, npoints, xmin, xmax,
                 ymin=None, ymax=None,
                 color=vp.color.red, 
                 bufsize=-1,
                 width=250, 
                 height=200, 
                 align='right', 
                 fast=True, 
                 graph=None):

        self.xmin = xmin
        self.xmax = xmax
        self.y = np.zeros(npoints, dtype=float)
        self.x = np.linspace(xmin, xmax, npoints + 1)
        self.x = (self.x[:-1] + self.x[1:])/2

        if type(graph) == type(None):

            if ymin == None:
                self.g = vp.graph(
                    title=title, xtitle=xtitle, ytitle=ytitle,
                    xmin=xmin, xmax=xmax, 
                    align=align, width=width, height=height,
                    fast=fast) # draw simpler graph for speed
            else:
                self.g = vp.graph(
                    title=title, xtitle=xtitle, ytitle=ytitle,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    align=align, width=width, height=height,
                    fast=fast) # draw simpler graph for speed      
            self.curve = vp.gcurve(graph=self.g, color=color)
        else:
            self.g = None
            self.curve = vp.gcurve(graph=graph, color=color)

        if bufsize > 0:
            self.buffer = CircularBuffer(bufsize, npoints)
            self.has_buffer = True
        else:
            self.buffer = None
            self.has_buffer = False
        
    def __del__(self):
        self.curve.delete()
        if self.g is not None:
            self.g.delete()
            
    def delete(self):
        self.__del__()
        
    def fill(self, y):   
        if self.has_buffer:
            self.buffer.append(y)
            oldest_y = self.buffer.get_oldest()
            if type(oldest_y) != type(None):
                self.y -= oldest_y
            self.y += y
        else:     
            self.y[:] = y

    def clear(self):
        self.y.fill(0)
        
    def draw(self):
        A = 1.0
        if self.has_buffer:
            A /= self.buffer.count

        self.curve.delete() # clear all points and replot
        for x, y in zip(self.x, A*self.y):
            self.curve.plot(x, y)

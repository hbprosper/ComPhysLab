# ----------------------------------------------------------------------
# Newton 2nd Law Utilities
# Created: Wed Oct 11 2023 Harrison B. Prosper
# ----------------------------------------------------------------------
import os, sys
import numpy as np

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from comphyslab.graphics import plot_central_xy_axes
from comphyslab.vectors import magnitude, norm

# update fonts
FONTSIZE = 12
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# use latex if available on system, otherwise set usetex=False
# module for shell utilities
import shutil

mp.rc('text', usetex=shutil.which('latex') is not None)

# use JavaScript for rendering animations
mp.rc('animation', html='jshtml')
# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
G = 6.674080e-11               # Gravitational constant (m^3 /kg /s^2)

Msun     = 1.98850e30          # Mass of Sun (kg)
Mmercury = 0.33010e24          # Mass of Mercury (kg)
Mvenus   = 4.86730e24          # Mass of Venus (kg)
Mearth   = 5.97220e24          # Mass of Earth (kg)
Mmars    = 0.64169e24          # Mass of Mars (kg)
Mjupiter = 1898.13e24          # Mass of Jupiter (kg)

Rsun     = 6.957e8             # Radius of Sun (m)
Rearth   = 6.371e6             # Radius of Earth (m)

# Conversion factors
DAY2SECS = 24*3600.0          # Seconds per Earth day
YEAR2SECS= 365.25*DAY2SECS    # Seconds per Earth year
AU2METERS= 1.495979e+11       # Astronomical unit (m)
DEG2RAD  = np.pi / 180        # need to convert angles to radians
DAY = DAY2SECS
YEAR= YEAR2SECS
AU  = AU2METERS
# ----------------------------------------------------------------------
# Given the masses `m` and positions `r` of the objects (planets, etc.), 
# compute the net gravitational force on each object by performing the 
# double loop over positions ourselves. 
# ----------------------------------------------------------------------
def compute_grav_forces1(m, r):
    
    # this list will contain the net forces acting on
    # each particle
    f = []
    
    # loop over the positions of the particles.
    # use the "enumerate" construct so that we 
    # have access to the index "i" of each position (i = 0, 1,..., n-1)
    
    for i, ri in enumerate(r):
        
        # "gi" will be the net gravitational field at the position
        # of particle i
        gi = np.zeros(3)
        
        # loop over the positions of all other particles
        for j, rj in enumerate(r):
            
            if i == j: continue  # skip self-interactions
            
            # by convention, we compute the vector that points
            # from particle "j" to particle "i"
            rij = ri - rj
            
            # find its magnitude, that is, the scalar distance
            # between particles "i" and "j"
            magrij = magnitude(rij)
            
            # sum the gravitational fields at the location of particle "i".
            gi += m[j] * rij / magrij**3
            
        gi = -G * gi
        
        # compute net force on particle "i"
        fnet = m[i] * gi
        
        # cache net force
        f.append(fnet)
    
    # convert list of forces to a numpy array of forces
    return np.array(f)
# ----------------------------------------------------------------------
# Given the masses m and positions r of the objects (planets, etc.), 
# compute the net force on each object, using the numpy broadcasting 
# mechanism. 
# ----------------------------------------------------------------------
def compute_forces(k, q, r, m, law=lambda x: 1/x**2):
    '''
    k (float)            Field strength. For electric fields this is 
                         the electric constant. For gravity k = -G.
                         
    q (array floats)     Charges. For electric fields this is an array
                         of electric charges. For gravity this is an
                         array of masses.
                         
    r (array of vectors) Position vectors, where each position vector is
                         modeled as a numpy array of shape (3,).
                         
    m (array of masses)

    law(x)               Function that compute force law [default 1/x**2]
    '''
    # Initial shapes of arrays q (charges) and r (position vectors) for
    # n particles
    # q.shape: (n, )
    # r.shape: (n, 3)
    # m.shape: (n, )
    
    ri = r[np.newaxis, :]    # change shape from (n, 3) to (1, n, 3)
    rj = r[:, np.newaxis]    # change shape from (n, 3) to (n, 1, 3)
    
    # Compute all possible vector differences using broadcasting
    # ----------------------------------------------------------
    # 1. The row (1, n, 3) with n columns, each element of which 
    #    is a vector, is replicated vertically n times to form 
    #    an array of shape (n, n, 3)
    #
    # 2. The column (n, 1, 3) with n rows, each element of which 
    #    is a vector, is replicated horizontally n times to form
    #    an array of shape (n, n, 3).
    #
    # 3. Now that we have arrays of the same shape, we can perform 
    #    element-by-element subtractions.
    #
    # Note: the vectors along the diagonal are all zero-length vectors!
    #       so self-interactions will be zero provided that we take
    #       care not to divide by zero.
    rij = ri - rj                         # shape: (n, n, 3)
     
    # rij_mag has shape (n, n) 
    rij_mag = magnitude(rij)              # shape: (n, n)
    
    # To avoid divide by zero for the terms with i=j,
    # replace the zeros along the diagonal with ones. 
    np.fill_diagonal(rij_mag, 1)
        
    # Why must we change the shape of the magnitudes array?
    rij_mag = rij_mag[:, :, np.newaxis]   # change shape to (n, n, 1)

    # As noted above, the self-interaction of particle "i" with itself 
    # is zero because the vector rij = 0 when i=j. 
    # Note: when i=j we have set rij_mag = 1.
    rij_hat = rij / rij_mag               # compute unit vectors
    pij = rij_hat * law(rij_mag)
      
    # Why must we change the shape of the charge array?
    # change shape from (n, ) to (n, 1, 1)
    qj = q[:, np.newaxis, np.newaxis] 
    
    # Compute the field of every particle j at the location 
    # of particle "i"
    Ei = k * qj * pij  # (n, 1, 1) x (n, n, 3) => (n, n, 3)
    
    # Compute the net field at the location of particle "i" by
    # summing the fields due to all other particles. Since the
    # particles are listed row-by-row, we must sum along axis 0, 
    # that is, "vertically".
    Ei = Ei.sum(axis=0)
    
    # Compute the net force per unit mass f = q E / m, acting 
    # on each particle
    qi = q[:, np.newaxis] # shape (n, 1)
    mi = m[:, np.newaxis] # shape (n, 1)
    a  = qi * Ei / mi     # shape: (n, 3)
    return a

def compute_grav_forces(m, r):
    return compute_forces(-G, m, r, m)
    
def compute_grav_forces2(m, r):
    
    # initial shapes of arrays m (masses) and r (position vectors)
    # m.shape: (n, )
    # r.shape: (n, 3)
    
    ri = r[np.newaxis, :]    # change shape from (n, 3) to (1, n, 3)
    rj = r[:, np.newaxis]    # change shape from (n, 3) to (n, 1, 3)
    
    # compute all possible vector differences using broadcasting:
    # 1. The row (1, n, 3) with n columns, each element of which 
    #    is a 3-vector, is replicated vertically n times to form 
    #    an array of shape (n, n, 3)
    #
    # 2. The column (n, 1, 3) with n rows, each element of which 
    #    is a 3-vector, is replicated horizontally n times to form
    #    an array of shape (n, n, 3).
    #
    # 3. Now that we have arrays of the same shape, we can perform 
    #    element-by-element subtractions.
    #
    # Note: the vectors along the diagonal are all zero-length vectors 
    # (see above)
    rij= ri - rj            # shape: (n, n, 3)
    
    # magrij has shape (n, n) 
    magrij = magnitude(rij)
    
    # For the magnitudes, replace the zeros along the diagonal
    # with ones. Why must we do this in this function but not
    # in compute_forces1?
    np.fill_diagonal(magrij, 1)
        
    # Why must we change the shape of the magnitudes array?
    magrij = magrij[:, :, np.newaxis]  # change shape to (n, n, 1)
    
    # Why must we change the shape of the mass array?
    mj = m[:, np.newaxis, np.newaxis]  # change shape from (n,) to (n,1,1)

    # For each particle, compute all gravitational fields
    gi = -G * mj * rij / magrij**3     # shape of f: (n, n, 3)
    
    # Compute net gravitational fields by summing the fields along 
    # axis 0, that is, "vertically".
    gi = gi.sum(axis=0)
    
    # Compute the net force acting on each particle 
    mi = m[:, np.newaxis]
    fnet = mi * gi # shape: (n, 3)
    
    return fnet
# ----------------------------------------------------------------------
class Missing:
    pass
MISSING = Missing()

class Solver:
    '''
    
    solver = Solver(forces, m, h)
    
    N = 1000
    orbits = [r0]                            # positions at t = 0 (i = 0)
    orbits.append( solver.compute(r0, v0) )  # positions at t = h (i = 1)
    
    for i in range(1, N-1):
        orbit = orbits.append( solver.compute(orbits[i]) ) 
	# positions at t = (i + 1) * h
        
    orbits = np.array(orbits)
    '''
    def __init__(self, forces, m, h):
        
        self.forces = forces  # function to compute net forces
        self.m  = m           # masses 
        self.h  = h           # time step
        self.hh = h**2
        
    def compute(self, r, v=MISSING):
        
        h, hh, m, forces = self.h, self.hh, self.m, self.forces
    
        if v is not MISSING:
            
            # use less precise formula at t = 0
            
            # compute the net forces at positions r
            F  = forces(m, r)
            
            # broadcast the mass over the vector components 
    	    # (see description below)
            Fm = F / m[:, np.newaxis]
            
            rnew = r + v * h + Fm * hh /  2   # O(h^3) accuracy
            
            self.r_prev = r  # needed in O(h^4) formula
            
        else:
            
            # Use more precise formula
            
            # Compute the net forces
            F   = forces(m, r)
            
            Fm  = F / m[:, np.newaxis]
            
            rnew= 2 * r - self.r_prev + Fm * hh 
         
            self.r_prev = r  # cache current positions
        
        return rnew
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
# ----------------------------------------------------------------------    

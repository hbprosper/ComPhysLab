# ----------------------------------------------------------------------
# Newton 2nd Law Utilities
# Created: Wed Oct 11 2023 Harrison B. Prosper
# ----------------------------------------------------------------------
import os, sys
import numpy as np
import h5py

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import qmc
from scipy.spatial import cKDTree

from comphyslab.vectors import magnitude, unit, dot
from comphyslab.utils import Bag, round_sig
# ----------------------------------------------------------------------
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
G  = 6.674080e-11              # Gravitational constant (m^3 /kg /s^2)
KB = 1.380649e-23              # Boltzmann constant (J/K)

Msun     = 1.98850e30          # Mass of Sun (kg)
Mmercury = 0.33010e24          # Mass of Mercury (kg)
Mvenus   = 4.86730e24          # Mass of Venus (kg)
Mearth   = 5.97220e24          # Mass of Earth (kg)
Mmars    = 0.64169e24          # Mass of Mars (kg)
Mjupiter = 1898.13e24          # Mass of Jupiter (kg)

Rsun     = 6.957e8             # Radius of Sun (m)
Rearth   = 6.371e6             # Radius of Earth (m)

# Conversion factors
DAY2SECS = 24*3600.0           # Seconds per Earth day
YEAR2SECS= 365.25*DAY2SECS     # Seconds per Earth year
AU2METERS= 1.495979e+11        # Astronomical unit (m)
DEG2RAD  = np.pi / 180         # need to convert angles to radians
DAY = DAY2SECS
YEAR= YEAR2SECS
AU  = AU2METERS
# ----------------------------------------------------------------------
# Force laws
# ----------------------------------------------------------------------
# Gravity
#------------------------------
def v_central_G(r):
    return -1.0 / r            # 1/r
    
def f_central_G(r):
    inv_r1 = 1.0 / r           # 1/r
    return inv_r1**2           # 1/r^2
    
def g_central_G(r):
    inv_r1 = 1.0 / r           # 1/r
    return -2.0 * inv_r1**3    # 1/r^3
    
Gravity = (v_central_G, f_central_G, g_central_G) 
#-------------------------------
# Lennard-Jones Force
#-------------------------------
def v_central_LJ(r): 
    inv_r1  = 1.0 / r     # 1/r
    inv_r2  = inv_r1**2   # 1/r^2
    inv_r6  = inv_r2**3   # 1/r^6
    inv_r12 = inv_r6**2   # 1/r^12
    return 4.0 * (inv_r12 - inv_r6)
    
def f_central_LJ(r): 
    inv_r1  = 1.0 / r     # 1/r
    inv_r2  = inv_r1**2   # 1/r^2
    inv_r6  = inv_r2**3   # 1/r^6
    inv_r12 = inv_r6**2   # 1/r^12
    return 24.0 * inv_r1 * (2 * inv_r12 - inv_r6)
    
def g_central_LJ(r):
    inv_r1  = 1.0 / r     # 1/r
    inv_r2  = inv_r1**2   # 1/r^2
    inv_r6  = inv_r2**3   # 1/r^6
    inv_r12 = inv_r6**2   # 1/r^12
    return 24.0 * inv_r2 * (-26.0 * inv_r12 + 7.0 * inv_r6)

TLennardJones = (v_central_LJ, f_central_LJ, g_central_LJ)
# ---------------------------------------------------------------------- 
# Given a central force law, compute the net force per unit mass field 
# at each particle. Code tidied up and corrected by ChatGPT 5.2.
# ----------------------------------------------------------------------
def compute_net_force_field(k, q, m, r, law, L=-1.0, eps=1e-16):
    """
    Arguments:
      k: scalar    field strength (e.g., electric constant or Newton's -G)
      q: (n,)      charges for n particles
      m: (n,)      masses
      r: (n,3)     particle positions
      law: (2,)    central force functions, f(r), g(r)
      L: scalar    (optional) bounding box for periodic boundary conditions
    
    Returns:
      f_net: (n,3)  force-per-unit-mass acting on each particle 
                    (i.e., acceleration field)
      
    Notes:
      Pairwise central field from j to i:  
          f_ij = k * q_j * f(r_ij) * rhat_ij
          
      Then particle acceleration is (q_i/m_i) * sum_j f_ij.
    """
    v_central, f_central, _ = law            # f(r)

    # Pairwise relative vectors: rij[i,j] = r_i - r_j
    rij = r[:, None, :] - r[None, :, :]      # (n,n,3)

    # Periodic minimum-image for cubic box
    if L > 0:
        rij -= L * np.round(rij / L)

    # Distances
    r2 = np.sum(rij * rij, axis=-1) + eps    # (n,n)
    np.fill_diagonal(r2, np.inf)
    rmag = np.sqrt(r2)                       # (n,n)

    # Unit vectors
    rhat = rij / rmag[..., None]             # (n,n,3)

    # Scalars f(r)
    fr = f_central(rmag)                     # (n,n)

    # Source charge must index j
    q_src = q[None, :, None]                 # (1,n,1)

    # Pair field from j to i (vector): Eij ~ q_j f(r) rhat
    Eij = k * q_src * (fr[..., None] * rhat) # (n,n,3)

    # Net field at each i: sum over sources j (axis=1)
    E = np.sum(Eij, axis=1)                  # (n,3)

    # Convert field to force-per-unit-mass: (q_i/m_i) * E
    qm = (q / m)[:, None]                    # (n,1)
    f_net = qm * E                           # (n,3)

    # Potential energy
    V = v_central(rmag)                      # (n,n), should give 0 on diag
    qq = q[:, None] * q[None, :]
    U = 0.5 * np.sum(k * qq * V)
    return f_net, U

def predictor_order3(r, v, f, h):
    h2 = h*h
    r1 = r + v*h + 0.5*f*h2
    v1 = v + f*h
    return r1, v1

def propagate_order3(k, q, m, r, v, law, h, L=-1.0):
    """
    Arguments:
      k: scalar    field strength (e.g., electric constant or Newton's -G)
      q: (n,)      charges for n particles
      m: (n,)      masses
      r: (n,3)     particle positions
      v: (n,3)     particle velocities
      law: (f, g)  central force functions, f(r) and g(r) = df/dr
      h: scalar    time step
      L: scalar    (optional) bounding box for periodic boundary conditions
                   assuming box is centered at the origin.
    
    Returns:
      rn1: (n,3)   predicted positions
      vn1: (n,3)   predicted velocities
      
    Notes:
      Pairwise central field from j to i:  
          f_ij = k * q_j * f(r_ij) * rhat_ij
          
      Then particle acceleration is (q_i/m_i) * sum_j f_ij.
    """
    # Evaluate at tn
    rn = r
    vn = v
    fn, U = compute_net_force_field(k, q, m, rn, law, L=L)

    # Predict positions and velocities
    r_star, v_star = predictor_order3(rn, vn, fn, h)

    # Evaluate field at predicted positions at tn+1 = tn + h
    f_star, _ = compute_net_force_field(k, q, m, r_star, law, L=L)

    # Correct velocity
    vn1 = vn + 0.5*h*(fn + f_star)                     # O(h^3)

    # Correct position (using corrected velocity)
    rn1 = rn + 0.5*h*(vn + vn1) + (h*h/12)*(fn-f_star) # O(h^3)

    # Optional: wrap positions after step for periodic
    # boundary conditions assuming a cubic box
    if L > 0:
        rn1 = (rn1 + 0.5*L) % L - 0.5*L

    return rn1, vn1, U

FCC = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])

CUBIC = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ])

def initialize_lattice(n_cells, basis, full=False, eps=1.e-6):
    """
    Build lattice in the unit box.
    
    n_cells : number of cells
    
    Returns:
        r  : (N,3) positions
    """

    # Build lattice
    a = 1.0 / n_cells  # FCC lattice constant

    lmin =-eps
    lmax = eps + 1.0

    n = 1 if full else 0
    
    r = []
    for i in range(n_cells+n):
        for j in range(n_cells+n):
            for k in range(n_cells+n):
                cell_origin = np.array([i, j, k]) * a
                for b in basis:
                    point = cell_origin + b * a
                    if full:
                        if (point > lmax).any(): 
                            continue
                        if (point < lmin).any(): 
                            continue
                    r.append(point)

    r = np.array(r)
    r -= 0.5
    return r

def initialize_fcc(n_cells, full=False, eps=1.e-3):
    """
    Build fcc lattice in the unit box.
    
    n_cells : number of cells
    
    Returns:
        r  : (N,3) positions
    """
    return initialize_lattice(n_cells, FCC, full, eps)

def initialize_cubic(n_cells, full=False, eps=1.e-3):
    """
    Build cubic lattice in the unit box.
    
    n_cells : number of cells
    
    Returns:
        r  : (N,3) positions
    """
    return initialize_lattice(n_cells, CUBIC, full, eps)

def initialize_sobol(n):
    """
    Build Sobol lattice in unit cube.
    
    n: number of particles given by N = 2**n
    
    Returns:
        r  : (N,3) positions
    """
    N = 2**n
    
    # Generate Sobol points in the unit cube and scale to bounds
    sampler= qmc.Sobol(d=3, scramble=False)
    M =  int(np.floor(np.log(N)/np.log(2)+0.01))
    r = sampler.random_base2(m=M)
    
    lower = 0.5*np.array([-1,-1,-1])
    upper = 0.5*np.array([ 1, 1, 1])
    
    r = qmc.scale(r, lower, upper).astype(np.float32)
    return r

def initialize_velocities(N, Tstar):
    # Maxwell velocities (Gaussian with variance = Tstar)
    v = np.random.normal(0.0, np.sqrt(Tstar), size=(N,3))

    # Remove center-of-mass velocity
    v -= np.mean(v, axis=0)
    return v

def pairwise_separations(r, L=-1):
    rij = r[:, None, :] - r[None, :, :]

    # Apply periodic boundary conditions if requested
    if L > 0:
        rij -= L * np.round(rij / L)

    # Distances for i<j only
    N  = r.shape[0]
    d  = np.linalg.norm(rij, axis=-1)
    iu = np.triu_indices(N, k=1)
    d  = d[iu]
    return d

def min_separation(r, L=None):
    """
    r: (n,3) array
    L: None  If scalar L, assumes cubic box [0,L) (or centered; 
       doesn't matter for distances).
    Returns: minimum pair distance (float)
    """
    tree = cKDTree(r, boxsize=L)  # boxsize enables periodic distances
    d, _ = tree.query(r, k=2)     # nearest neighbor is itself (k=1), so take k=2
    return d[:, 1].min()
    
def maxwell_distribution(T, vmin, vmax, nbins=50):
    '''
    T          dimensionless (T = mass * v0**2 / KB to convert to K)
    vmin, vmax 
    nbins
    '''
    dv = (vmax-vmin)/nbins
    beta = 1/T
    b  = 4*np.pi*np.sqrt((beta/2/np.pi)**3)
    v  = np.linspace(vmin, vmax, nbins+1)
    v  = 0.5*(v[:-1] + v[1:])
    vv = v*v
    y  = b * vv * np.exp(-beta * vv/2) * dv
    return y, v
    
def radial_distribution(rho, r, rmax, nbins=50, rcore=None, L=None):
    """
    Compute g(r) for a configuration of particles. If L is not None
    apply periodic boundary conditions in a cubic box of side length
    L.

    Arguments:
        rho  : density (particles per unit volume)
        r    : (N,3) particle positions
        rmax : maximum inter-particle distance
        nbins: number of radial bins
        rcore: If specified, use only those particles within
               this distance of the origin.
        L    : If L is not None, apply cubical boundary conditions

    Returns:
        rc   : bin centers
        g    : g(r) values
    """

    # If rcore specified, use only those particles within
    # rcore of the origin.
    if rcore is not None:
        radii = np.linalg.norm(r, axis=-1)
        p = r[radii < rcore]
    else:
        p = r
    
    # Compute pairwise displacements
    rij = p[:, None, :] - p[None, :, :]

    # Apply periodic boundary conditions if requested
    if L is not None:
        rij -= L * np.round(rij / L)

    # Distances for i<j only
    d = np.linalg.norm(rij, axis=-1)
    N = p.shape[0]
    I = np.triu_indices(N, k=1)
    d = d[I]

    # Histogram the distances
    edges = np.linspace(0.0, rmax, nbins + 1)
    counts, _ = np.histogram(d, bins=edges)

    # Convert pair counts to g(r)
    # For an ideal gas, the density is uniform, so the
    # expected number of pairs in a shell for an ideal gas
    # is given by:
    #   N * rho * shell_vol / 2  (divide by 2 because we 
    # used i<j pairs)
    rin, rout = edges[:-1], edges[1:]
    shell_vol = (4.0/3.0) * np.pi * (rout**3 - rin**3)
    ideal_gas_counts = N * rho * shell_vol / 2
    
    g  = counts / ideal_gas_counts
    rc = 0.5 * (rin + rout) # Bin centers
    
    return g, rc

def radial_density_profile(r, R, nbins=50):
    """
    Returns bin centers rc and density rho(rc) in spherical shells.
    """
    d = np.linalg.norm(r, axis=1)
    edges = np.linspace(0.0, R, nbins + 1)
    counts, _ = np.histogram(d, bins=edges)

    rin, rout = edges[:-1], edges[1:]
    rc = 0.5 * (rin + rout)
    
    shell_vol = (4.0/3.0) * np.pi * (rout**3 - rin**3)
    rho = counts / shell_vol

    return rho, rc
    
class WallPressure:
    def __init__(self, R, m=1.0):
        self.R = float(R)
        self.A = 4.0 * np.pi * self.R**2
        self.m = float(m)
        self.impulse_sum = 0.0  # sum of normal impulses over a window

    def add_impulses(self, vdotn_outward):
        # vdotn_outward: array of outward normal speeds at impact (positive)
        self.impulse_sum += np.sum(2.0 * self.m * vdotn_outward)

    def pressure(self, dt_window):
        # dt_window: elapsed time since last pressure readout
        return self.impulse_sum / (self.A * dt_window)

    def reset(self):
        self.impulse_sum = 0.0
        
# ----------------------------------------------------------------------
# 4th-order Hermite predictor-corrector
# ----------------------------------------------------------------------
def compute_acceleration_and_jerk_fields(k, q, m, r, v, law, L=-1.0, eps=1e-16):
    """
    Arguments:
      k: scalar    field strength (e.g., electric constant or Newton's -G)
      q: (n,)      charges for n particles
      m: (n,)      masses
      r: (n,3)     particle positions
      v: (n,3)     particle velocities
      law: (f, g)  central force functions, f(r) and g(r) = df/dr
      L: scalar    (optional) bounding box for periodic boundary conditions
    
    Returns:
      f_net: (n,3)  force-per-unit-mass acting on each particle 
                    (i.e., acceleration field)
                    
      j_net: (n,3)  time-derivative of f_net
      
    Notes:
      Pairwise central field from j to i:  
          f_ij = k * q_j * f(r_ij) * rhat_ij
          
      Then particle acceleration is (q_i/m_i) * sum_j f_ij.
    """
    v_central, f_central, g_central = law    # V(r), f(r), df/dr

    # Pairwise relative vectors: rij[i,j] = r_i - r_j
    rij = r[:, None, :] - r[None, :, :]      # (n,n,3)
    uij = v[:, None, :] - v[None, :, :]      # (n,n,3)

    # Periodic minimum-image for cubic box
    if L > 0:
        rij -= L * np.round(rij / L)

    # Distances
    r2 = np.sum(rij * rij, axis=-1) + eps    # (n,n)
    np.fill_diagonal(r2, np.inf)
    rmag = np.sqrt(r2)                       # (n,n)

    # Unit vectors
    rhat = rij / rmag[..., None]             # (n,n,3)

    # Scalars f(r), g(r)=df/dr
    fr = f_central(rmag)                     # (n,n)
    gr = g_central(rmag)                     # (n,n)

    # Source charge must index j
    q_src = q[None, :, None]                 # (1,n,1)

    # Pair field from j to i (vector): Eij ~ q_j f(r) rhat
    Eij = k * q_src * (fr[..., None] * rhat) # (n,n,3)

    # Build jerk field Jij = d/dt Eij (uses relative velocity uij)
    rhat_dot_u = np.sum(rhat * uij, axis=-1)                 # (n,n)
    u_r = (rhat_dot_u[..., None]) * rhat                     # (n,n,3)
    u_t = uij - u_r                                          # (n,n,3)

    # d/dt [ f(r) rhat ] = g(r) * (dr/dt) * rhat + f(r)/r * u_t
    # and dr/dt = rhat·u, so g(r)*(rhat·u)*rhat = g(r)*u_r (vector)
    Jij = k * q_src * (gr[..., None] * u_r + (fr / rmag)[..., None] * u_t)

    # Net field at each i: sum over sources j (axis=1)
    E = np.sum(Eij, axis=1)                  # (n,3)
    J = np.sum(Jij, axis=1)                  # (n,3)

    # Convert field to force-per-unit-mass: (q_i/m_i) * E
    qm = (q / m)[:, None]                    # (n,1)
    f_net = qm * E                           # (n,3)
    j_net = qm * J                           # (n,3)

    # Potential energy
    V = v_central(rmag)                      # (n,n), should give 0 on diag
    qq = q[:, None] * q[None, :]
    U = 0.5 * np.sum(k * qq * V)
    
    return f_net, j_net, U

def predictor4(r, v, f, j, h):
    h2 = h*h
    h3 = h2*h
    r1 = r + v*h + 0.5*f*h2 + (1/6)*j*h3
    v1 = v + f*h + 0.5*j*h2
    return r1, v1

def propagate_order4(k, q, m, r, v, law, h, L=-1.0):
    """
    Arguments:
      k: scalar    field strength (e.g., electric constant or Newton's -G)
      q: (n,)      charges for n particles
      m: (n,)      masses
      r: (n,3)     particle positions
      v: (n,3)     particle velocities
      law: (f, g)  central force functions, f(r) and g(r) = df/dr
      h: scalar    time step
      L: scalar    (optional) bounding box for periodic boundary conditions
    
    Returns:
      rn1: (n,3)   predicted positions
      vn1: (n,3)   predicted velocities

    Notes:
      Pairwise central field from j to i:  
          f_ij = k * q_j * f(r_ij) * rhat_ij
          
      Then particle acceleration is (q_i/m_i) * sum_j f_ij.
    """
    # Evaluate at tn
    rn = r
    vn = v
    fn, jn, U = compute_acceleration_and_jerk_fields(
        k, q, m, rn, vn, law, L=L)

    # Predict
    r_star, v_star = predictor4(rn, vn, fn, jn, h)

    # Evaluate at predicted positions at tn+h
    f_star, j_star, _ = compute_acceleration_and_jerk_fields(
        k, q, m, r_star, v_star, law, L=L)

    h2 = h*h

    # Correct velocity
    v_np1 = vn + 0.5*h*(fn + f_star) + (h2/12)*(jn - j_star)

    # Correct position (using corrected velocity)
    r_np1 = rn + 0.5*h*(vn + v_np1) + (h2/12)*(fn - f_star)

    # Optional: wrap positions after the step for periodic
    # boundary conditions
    if L > 0:
        r_np1 = (r_np1 + 0.5*L) % L - 0.5*L

    return r_np1, v_np1, U
# ----------------------------------------------------------------------
class Missing:
    pass
MISSING = Missing()

class CentralForceSolver:
    '''
    solver = CentralForceSolver(k, q, m, r, v, law, h, nsteps)
    solver.compute()
    '''
    def __init__(self, k, q, m, r, v, law, h, nsteps):        
        self.k = k
        self.q = q
        self.m = m           # masses
        self.r = r
        self.v = v
        self.law = law
        self.h = h           # time step
        self.nsteps = nsteps
        
    def compute(self):
        R = []
        V = []
        for _ in range(self.nsteps):
            r, v, U = propagate_order3(
                self.k, self.q, self.m, self.r, self.v, self.law, self.h)
            R.append(r)
            V.append(v)
        return np.array(R), np.array(V)
# ---------------------------------------------------------------------- 
# Argon parameters
# ---------------------------------------------------------------------- 
def argon_initial_state(rho, T):   
    sigma   = 3.4e-10                 # Distance at which potential is zero (m)
    epsilon = 1.657e-21               # Characteristic energy (J)
    mass    = 6.69e-26                # Mass of argon atom (kg)
    equi_sep= 2**(1/6)                # Equilibrium separation (sigma)
    vc = float(np.sqrt(epsilon/mass)) # Characteristic speed (m/s)
    tc = float(sigma / vc)            # Characteristic timescale (s)

    # Save in a bag
    bg = Bag()
    bg.sigma   = round_sig(sigma)
    bg.epsilon = round_sig(epsilon)
    bg.mass    = round_sig(mass)
    bg.equi_sep= round_sig(equi_sep)
    bg.vc      = round_sig(vc)
    bg.tc      = round_sig(tc)
    bg.T2K     = round_sig(mass * vc**2 / KB)

    print(f'''
    sigma:       {bg.sigma:10.2e} m
    epsilon:     {bg.epsilon:10.2e} J
    mass:        {bg.mass:10.2e} kg
    speed scale: {bg.vc:5.2f} m/s
    time scale:  {bg.tc:10.2e} s
    ''')

    # -------------------------------------------------------
    # Compute number density (units atoms/sigma**3)
    # -------------------------------------------------------
    print(f'requested number density: {rho:10.3e} kg/m^3')
    
    rho = rho / mass           # number of atoms/m^3
    print(f'requested number density: {rho:10.3e} atoms/m^3')
    
    rho = rho * sigma**3       # number of atoms/sigma**3
    bg.rho = round_sig(rho)
    print(f'requested number density: {rho:10.3e} atoms/sigma^3')

    bg.T = round_sig(T)
    print(f'requested temperature:    {T:6.3f} K\n')
    # -------------------------------------------------------
    # Particles
    # -------------------------------------------------------    
    # Create a lattice of points that fit within a sphere
    # such that 1) the points are as far apart as possible and
    # 2) the average particle density equals the requested
    # density.
    # -------------------------------------------------------
    # Step 1. Create a unit cube of lattice points centered at the
    # origin.
    ncells = 4
    r  = initialize_fcc(ncells, full=True) 

    # Step 2: Scale the unit cube of lattice points by the 
    # length "L" so that the average point density is 2*rho.
    N1 = len(r)
    V1 = N1/(2*rho)
    L  = V1**(1/3)
    r *= L
    print(f'number of lattice points generated: {N1}\n')

    # Step 3: Keep those lattice points that lie within 
    # a sphere of radius L/sqrt(2)
    rmag = magnitude(r)
    bg.r = r[rmag < L/np.sqrt(2)]
    bg.N = len(bg.r)
   
    # Step 4: Compute radius of the circumscribing sphere such 
    # that the particle density equals the requested density.
    R = (3*bg.N/rho/4/np.pi)**(1/3)    
    bg.R = R
    print(f'number atoms:   {bg.N:5d} atoms     (container radius, R = {bg.R:6.3f} sigma)')

    # Sanity check!
    V = (4/3)*np.pi*R**3
    bg.rho = bg.N / V
    print(f'number density:    {bg.rho:10.3e}/sigma^3')

    rmin_sep = round_sig(min_separation(bg.r))
    bg.rmin_sep = rmin_sep
    print(
        f'min(separation):   {rmin_sep:6.3f} sigma '\
        f'(cf. {equi_sep:6.3f} sigma)\n')

    # -------------------------------------------------------
    # Generate initial velocities
    # -------------------------------------------------------
    vrms1 = np.sqrt(3*KB*T/mass)
    v = vrms1 * unit(np.random.uniform(-1.0, 1.0, 3*bg.N).reshape(bg.N, 3))
    
    # Remove center of mass motion and rescale velocities 
    # to arrive at specified temperature, defined as
    # T = (1/3)(m/KB) <v^2>
    if (len(v.shape) > 1) and (v.shape[0] > 1):
        v -= v.mean(axis=0)
    vrms2 = np.sqrt(np.mean((v**2).sum(axis=-1)))
    v *= vrms1 / vrms2
    bg.v = v
    
    # Check that we get the requested temperature
    vrms = float(np.sqrt(np.mean((v**2).sum(axis=-1))))
    T = (1/3)*(mass/KB)*vrms**2
    print(f'Vrms:  {vrms:5.1f} m/s,\tT: {T:8.2f} K')

    # Compute dimensionless velocities and dimensionless
    # temperature
    v /= vc
    vrms /= vc
    T_reduced = float(np.mean((v**2).sum(axis=-1)) / 3)
    print(f'Vrms:  {vrms:5.1f} vc,\tT: {T_reduced:8.2f}')
    bg.vrms = round_sig(vrms)

    # Reserve buffers (used in update)
    bg.r0 = np.zeros_like(bg.r)
    bg.v0 = np.zeros_like(bg.v)

    bg.m  = np.ones(bg.N)      # Masses of particles (units of mass)
    bg.q  = np.ones(bg.N)      # Masses of atoms (units of mass)
    # --------------------------------------------
    # Force
    # --------------------------------------------
    bg.k = 1.0
    bg.law = TLennardJones
    return bg
# ----------------------------------------------------------------------    

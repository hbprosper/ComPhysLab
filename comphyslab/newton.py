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
G = 6.674080e-11               # Gravitational constant (m^3 /kg /s^2)
KB = 1.380_649e-23             # Boltzmann constant (J/K)

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
# Force laws
# ----------------------------------------------------------------------
# Gravity
#------------------------------
def f_central_G(r):
    inv_r1 = 1.0 / rs          # 1/r
    return inv_r1**2           # 1/r^2
    
def g_central_G(r):
    inv_r1 = 1.0 / rs          # 1/r
    return -2.0 * inv_r1**3    # 1/r^3
    
Gravity = (f_central_G, g_central_G) 
#-------------------------------
# Lennard-Jones Force
#-------------------------------
def f_central_LJ(r, R=np.inf):
    mask = r < R
    rs = r[mask]
    
    inv_r1  = 1.0 / rs    # 1/r
    inv_r2  = inv_r1**2   # 1/r^2
    inv_r6  = inv_r2**3   # 1/r^6
    inv_r12 = inv_r6**2   # 1/r^12

    out = np.zeros_like(r)
    out[mask] = 24.0 * inv_r1 * (2 * inv_r12 - inv_r6)
    return out
    
def g_central_LJ(r, R=np.inf):
    mask = r < R
    rs = r[mask]
    
    inv_r1  = 1.0 / rs    # 1/r
    inv_r2  = inv_r1**2   # 1/r^2
    inv_r6  = inv_r2**3   # 1/r^6
    inv_r12 = inv_r6**2   # 1/r^12
    
    out = np.zeros_like(r)
    out[mask] = 24.0 * inv_r2 * (-26.0 * inv_r12 + 7.0 * inv_r6)
    return out

TLennardJones = (f_central_LJ, g_central_LJ)
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
    f_central = law[0]                       # f(r)

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

    return f_net

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
    fn = compute_net_force_field(k, q, m, rn, law, L=L)

    # Predict positions and velocities
    r_star, v_star = predictor_order3(rn, vn, fn, h)

    # Evaluate field using predicted positions at tn+1 = tn + h
    f_star = compute_net_force_field(k, q, m, r_star, law, L=L)

    # Correct velocity
    vn1 = vn + 0.5*h*(fn + f_star)                     # O(h^3)

    # Correct position (using corrected velocity)
    rn1 = rn + 0.5*h*(vn + vn1) + (h*h/12)*(fn-f_star) # O(h^3)

    # Optional: wrap positions after step for periodic
    # boundary conditions assuming a cubic box
    if L > 0:
        rn1 = (rn1 + 0.5*L) % L - 0.5*L

    return rn1, vn1

FCC = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])

CUBIC = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ])

def initialize_lattice(n_cells, basis, full=False, eps=1.e-3):
    """
    Build lattice in the unit box.
    
    n_cells : number of cells
    
    Returns:
        r  : (N,3) positions
    """

    # Build lattice
    a = 1.0 / n_cells  # FCC lattice constant

    lmin = -eps*a
    lmax =  eps*a + 1.0

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
    L: None  If scalar L, assumes cubic box [0,L) (or centered; doesn't matter for distances).
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
    f_central, g_central = law               # f(r), df/dr

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

    return f_net, j_net

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
    fn, jn = compute_acceleration_and_jerk_fields(
        k, q, m, rn, vn, law, L=L)

    # Predict
    r_star, v_star = predictor4(rn, vn, fn, jn, h)

    # Evaluate at predicted positions at tn+h
    f_star, j_star = compute_acceleration_and_jerk_fields(
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

    return r_np1, v_np1
# ----------------------------------------------------------------------
class Missing:
    pass
MISSING = Missing()

class MDLoggerH5:
    def __init__(self, filename, N, dt, R, rho,
                 save_every=10, dtype=np.float32):
        self.f = h5py.File(filename, "w")
        self.N = int(N)
        self.dt = float(dt)
        self.R = float(R)
        self.rho = float(rho)
        self.save_every = int(save_every)
        self.dtype = dtype

        # --- metadata ---
        self.f.attrs["N"] = self.N
        self.f.attrs["dt"] = self.dt
        self.f.attrs["R"] = self.R
        self.f.attrs["rho"] = self.rho
        self.f.attrs["units"] = "LJ_reduced (m=kB=sigma=epsilon=1)"
        self.f.attrs["save_every"] = self.save_every

        # chunk a few frames at a time
        chunk_t = 16

        self.f.create_dataset(
            "r", shape=(0, self.N, 3), maxshape=(None, self.N, 3),
            dtype=dtype, chunks=(chunk_t, self.N, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,  # arrange bytes so that identical bytes are adjacent
            scaleoffset=4  # 4 sig figs after decimal point
        )
        self.f.create_dataset(
            "v", shape=(0, self.N, 3), maxshape=(None, self.N, 3),
            dtype=dtype, chunks=(chunk_t, self.N, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True, 
            scaleoffset=4
        )
        # impulse accumulated over the save interval (10 steps)
        self.f.create_dataset(
            "impulse", shape=(0,), maxshape=(None,),
            dtype=dtype, chunks=(4096,),
            compression="gzip",
            compression_opts=4,
            shuffle=True,     
            scaleoffset=4
        )

    def append(self, r, v, impulse):
        i = self.f["r"].shape[0]
        self.f["r"].resize(i + 1, axis=0)
        self.f["v"].resize(i + 1, axis=0)
        self.f["impulse"].resize(i + 1, axis=0)

        self.f["r"][i] = r.astype(self.dtype, copy=False)
        self.f["v"][i] = v.astype(self.dtype, copy=False)
        self.f["impulse"][i] = np.asarray(impulse, dtype=self.dtype)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()
        
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
            r, v = propagate_order3(
                self.k, self.q, self.m, self.r, self.v, self.law, self.h)
            R.append(r)
            V.append(v)
        return np.array(R), np.array(V)
# ----------------------------------------------------------------------    

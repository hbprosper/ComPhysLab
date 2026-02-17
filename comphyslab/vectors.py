# ----------------------------------------------------------------------
# Vector utilities based on numpy
# Created: Wed Oct 25 2023 Harrison B. Prosper
# ----------------------------------------------------------------------
import os, sys
import numpy as np
# ----------------------------------------------------------------------
# The following functions can operate on an array of vectors
# ----------------------------------------------------------------------
# Compute magnitudes of one or more vectors
def magnitude(v):
    '''
    Compute magnitude of a vector (modeled as numpy array) or an numpy
    array of vectors.
    '''
    return np.sqrt(np.sum(v*v, axis=-1))

# Compute unit vectors from one or more vectors
def norm(v):
    '''
    Compute the normalized vectors, that is, unit vectors given one or
    more vectors v.
    '''
    magv = np.sqrt(np.sum(v*v, axis=-1))
    magv = np.where(magv < 1.e-15, 1, magv) # handle zero-length vectors
    try: 
        u = v / magv[:, np.newaxis]
    except:
        u = v / magv
    return u

def dot(a, b):
    '''
    
    Compute the dot product of vectors a and b or an array of vectors 
    a and b, where the vectors are modeled as numpy arrays.
    
    Arguments
    ---------
    
    a: a vector or array of vectors
    b: a vector (modeled as a numpy array) or an numpy array of vectors
    
    Return
    ------
    
    ab: dot product
    
    Example
    -------
    
    ab = dot(a, b)
    
    '''
    try:
        c = (a*b).sum(axis=1)
    except:
        c = np.dot(a, b)
    return c

def tangent(u, n):
    '''
    Given the incident unit vector u and normal unit vector n, return 
    the unit vector n x u x n / |u x n| that is at right angles to n 
    and which lies in the plane defined by u and n.
    
    Arguments
    ---------
    u:    unit vector in direction of incident a ray (or an numpy array 
	      of unit vectors)
    n:    unit normal to boundary (or an numpy array of unit normals)
    
    Return
    ------
    nt:   unit vector in direction of tangent to normal (or a numpy 
	      array thereof)
    
    Example
    -------
    
    nt = tangent(u, n)
    '''
    return norm(np.cross(n, np.cross(u, n)))

def reflection(u, n):
    '''
    Given the incident unit vector u, normal unit vector n that defines 
    the orientation of the boundary between two media, return the unit 
    vector in the direction of the reflected ray.  
    
    Arguments
    ---------
    u:    Unit vector in direction of incident a ray (or a numpy array 
	      of unit vectors)
    n:    Unit normal to boundary (or an numpy array of unit normals)
    
    Return
    ------
    ur:   Unit vector in direction of reflected ray (or a numpy array 
	      thereof)
    
    Example
    -------
    
    ur = reflection(u, n)
    '''
    udotn = dot(u, n)
    try:
        udotn = udotn[:, np.newaxis] # why do we try this?
    except:
        pass
    return u - 2*udotn*n

def transmission(u, n, n1, n2):
    '''
    Given the incident unit vector u, normal unit vector n that defines 
    the orientation of the boundary between two media of refractive 
    indices n1 and n2, return the unit vector in the direction of the 
    transmitted ray.  
    
    Arguments
    ---------
    u:    unit vector in the direction of the incident ray (or an 
          numpy array of unit vectors)
    n:    unit normal to boundary (or an numpy array of unit normals)
    n1:   refractive index of medium traversed by incident ray
    n2:   refractive index of medium traversed by transmitted (i.e., 
          refracted) ray
    
    Return
    ------
    ut:   unit vector in direction of transmitted ray 
          (or a numpy array thereof)
    
    Example
    -------
    
    ut = transmission(u, n, n1, n2)
    '''
    
    # n x u x n
    nun   = np.cross(n, np.cross(u, n))
    
    udotn = dot(u, n)
    
    n12   = n1/n2
    
    q = 1-n12**2*(1-udotn**2)
    
    # protect against negative values.
    q = np.sqrt(np.where(q < 0, 0, q))
    
    scale = np.sign(udotn) * q
    
    try:
        scale = scale[:, np.newaxis] # again, why try this?
    except:
        pass
 
    try:
        n12 = n12[:, np.newaxis]
    except:
        pass
    
    return scale * n + n12 * nun

def line_sphere_intersect(c, u, a, o):
    '''
    
    Given a line defined by the point c and unit vector u, 
    compute the points of intersection with a sphere of radius a 
    located at point o.
    
    Arguments
    ---------
    c :   a point on the incident ray (or a numpy array of vectors)
    u :   a unit vector in the direction of the incident ray 
          (or a numpy array of vectors)
    a :   the radius of the sphere
    o :   the location of center of the sphere 
          (i.e., the center of curvature)
    
    Return
    ------
    p1, p2, crosses : p1 and p2 are the intersection points, with p1 
    the closer of the two points to point c and crosses are an array 
    of booleans. If True, the line crosses the sphere.
    
    Example
    -------
    
    p1, p2, crosses = line_sphere_intersect(C, U, R, O)
    '''
    if not isinstance(o, np.ndarray):
        raise TypeError(f'''
    The position vector of the center of the sphere must be
    a numpy array not of type {type(o)}
        ''')

    C  = c - o
    cc = dot(C, C)
    uc = dot(u, C)
    
    try:
        uc = uc[:, np.newaxis]
        cc = cc[:, np.newaxis]
    except:
        pass
    
    # solutions for lambda
    s = uc**2 - cc + a**2
    
    # check for valid solutions
    crosses = s > 0

    q  = np.sqrt(np.where(s < 0, 0, s))
    
    l1 =-uc + q
    l2 =-uc - q

    # First order intersection points according
    # value of l1 and l2 such that 
    lmin = np.where(l1 < l2, l1, l2)
    r1 = lmin * u + c
    
    lmax = np.where(l1 < l2, l2, l1)
    r2 = lmax * u + c

    return r1, r2, crosses

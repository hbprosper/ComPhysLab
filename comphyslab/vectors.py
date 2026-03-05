# ----------------------------------------------------------------------
# Vector utilities based on numpy
# Created: Wed Oct 25 2023 Harrison B. Prosper
# Updated: Sun Feb 22 2026 HBP: improve
# ----------------------------------------------------------------------
import os, sys
import numpy as np
# ----------------------------------------------------------------------
# The following functions can operate on an array of vectors
# ----------------------------------------------------------------------
def magnitude(v):
    '''
    Compute magnitude of a vector (modeled as a numpy array) or a numpy
    array of vectors.
    '''
    return np.sqrt((v*v).sum(axis=-1))

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
    return (a*b).sum(axis=-1)

def unit(v, eps=1.e-15):
    '''
    Compute unit vectors given one or more vectors v.
    '''
    is2D = len(v.shape) > 1
    magv = np.clip(magnitude(v), eps, None) # handle zero-length vectors
    return v / magv[:, None] if is2D else v / magv

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
    return unit(np.cross(n, np.cross(u, n)))

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
    
    ur = reflect(u, n)
    '''
    is2D  = len(u.shape) > 1 
    udotn = dot(u, n)
    return u - 2*udotn[:, None]*n if is2D else u - 2*udotn*n

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
    is2D = len(u.shape) > 1
    
    # n x u x n
    nun   = np.cross(n, np.cross(u, n))
    
    udotn = dot(u, n)
    
    n12   = n1/n2
    
    q = 1-n12**2*(1-udotn**2)
    
    # protect against negative values.
    q = np.sqrt(np.clip(q, 0, None))
    
    scale = np.sign(udotn) * q

    return scale[:, None] * n + n12[:, None] * nun if is2D else scale * n + n12

# def line_sphere_intersect(c, u, d, o):
#     '''
#     Given a line defined by the point c and unit vector u, 
#     compute the points of intersection with a sphere of radius d 
#     located at point o.
    
#     Arguments
#     ---------
#     c :   a point on the incident ray (or a numpy array of vectors)
#     u :   a unit vector in the direction of the incident ray 
#           (or a numpy array of vectors)
#     d :   the radius of the sphere
#     o :   the location of center of the sphere 
#           (i.e., the center of curvature)
    
#     Return
#     ------
#     p1, p2, crosses : p1 and p2 are the intersection points, with p1 
#     the closer of the two points to point c and crosses are an array 
#     of booleans. If True, the line crosses the sphere.
    
#     Example
#     -------
    
#     p1, p2, crosses = line_sphere_intersect(C, U, R, O)
#     '''
#     if not isinstance(o, np.ndarray):
#         raise TypeError(f'''
#     The position vector of the center of the sphere must be
#     a numpy array not of type {type(o)}
#         ''')

#     is2D = len(c.shape) > 1 
    
#     C  = c - o
#     cc = dot(C, C)
#     uc = dot(u, C)
    
#     # If the line crosses the sphere, s >= 0
#     s = uc**2 - cc + d**2
#     crosses = s >= 0
 
#     # Two possible intersection points
#     q = np.sqrt(np.maximum(s, 0))
#     t1 =-uc + q
#     t2 =-uc - q

#     # Order intersection points according
#     # values of t1 and t2  
#     tmin = np.where(t1 < t2, t1, t2)[:, None] if is2D \
#     else np.where(t1 < t2, t1, t2)
#     r1 = c + tmin * u
    
#     tmax = np.where(t1 < t2, t2, t1)[:, None] if is2D \
#     else np.where(t1 < t2, t2, t1)
#     r2 = c + tmax * u

#     # Return so that r1 contains points with the smaller 
#     # values of t and r2 the larger values of t

#     return r1, r2, crosses

def line_sphere_intersect(c, u, d, o=np.array([0.0, 0.0, 0.0])):
    '''
    Given a line defined by the point c and unit vector u, 
    compute the points of intersection with a sphere of radius d 
    located at point o.
    
    Arguments
    ---------
    c :   a point on the incident ray (or a numpy array of vectors)
    u :   a unit vector in the direction of the incident ray 
          (or a numpy array of vectors)
    d :   the radius of the sphere
    o :   the location of center of the sphere 
          (i.e., the center of curvature) (default np.array([0.0,0.0,0.0]))
    
    Return
    ------
    p1, p2, crosses : p1 and p2 are the intersection points, with p1 
    the closer of the two points to c if c is outside the sphere
    or the point in the direction of u if c is inside the sphere.

    crosses is an array of booleans. If True, the line crosses the sphere.
    
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
    
    # If the line crosses the sphere, s >= 0
    s = uc**2 - cc + d**2
    crosses = s >= 0
 
    # Two possible intersection points defined by the two
    # scalars t1 and t2
    q = np.sqrt(np.maximum(s, 0))
    t1 =-uc + q
    t2 =-uc - q

    # We need to decide which scalars to use for the
    # first and second points.
    l1 = np.zeros_like(t1) 
    l2 = np.zeros_like(t2)
    R  = magnitude(C) # distance from center of sphere
    
    # Handle points within sphere.
    # Choose the greater of t1 and t2  
    # --------------------------------
    inside = R < d
    if inside.sum() > 0:
        t1_inside  = t1[inside]
        t2_inside  = t2[inside]
     
        l1[inside] = np.where(t1_inside >  t2_inside, t1_inside, t2_inside)
        l2[inside] = np.where(t1_inside <= t2_inside, t1_inside, t2_inside)

    # Handle points outside sphere.
    # Choose the smaller of t1 and t2  
    # --------------------------------
    outside = R >= d
    if outside.sum() > 0:
        t1_outside = t1[outside]
        t2_outside = t2[outside]
    
        l1[outside] = np.where(t1_outside <  t2_outside, t1_outside, t2_outside)
        l2[outside] = np.where(t1_outside >= t2_outside, t1_outside, t2_outside)

    # Finally, compute points!
    many = len(c.shape) > 1 
    r1 = c + l1[:, None] * u if many else c + l1 * u
    r2 = c + l2[:, None] * u if many else c + l2 * u 

    return r1, r2, crosses

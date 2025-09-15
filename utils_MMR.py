"""
Multimodal registration utils
"""

import numpy as np
import math
from sympy import *


################################ point cloud functions ################################

def createPcCirc(z, radius, grid_size):
    """
    Doc:
    Creates a circular point cloud
    """
    
    # Calculate angle of points
    
    area = 2*math.pi*radius
    n = area/grid_size
    d_phi = 2*math.pi/n
    Phi = [i*d_phi for i in range(int(n) + 1)]
    
    # Create in x-y plane
    
    X = [radius*math.cos(phi) for phi in Phi]
    Y = [radius*math.sin(phi) for phi in Phi]
    pc_circ = np.empty((len(X), 3))
    for i in range(len(X)):
        pc_circ[i, :] = np.array([X[i], Y[i], z])
    
    return pc_circ


def createPcCylinder(length, radius, grid_size):
    '''
    Doc:
    Creates a cylindrical point cloud
    '''
    
    z_vals = np.arange(0, length, grid_size)
    pc_cyl = np.empty((0, 3))
    for ax_pt in z_vals:
        pc_circ = createPcCirc(z=ax_pt,
                               radius=radius,
                               grid_size=grid_size)
        pc_cyl = np.vstack((pc_cyl, pc_circ))

    return pc_cyl


def createPcHemisphere(radius, grid_size):
    '''
    Doc:
    Creates a hemisphere point cloud
    '''
    z_vals = np.arange(0, radius, grid_size)
    
    pc_hem = np.empty((0, 3))
    for z in z_vals:
        r = (radius**2 - z**2)**0.5
        pc_circ = createPcCirc(z=-z, 
                              radius=r,
                              grid_size=grid_size)
        pc_hem = np.vstack((pc_hem, pc_circ))
    
    return pc_hem


def create_LUS_pc(radius, grid_size):
    """"
    Doc:
    Creates LUS point cloud by spherocylindrical model
    """
    
    cyl_pc = createPcCylinder(length=60,
                              radius=radius,
                              grid_size=grid_size)
    hem_pc = createPcHemisphere(radius=radius, 
                                grid_size=grid_size)

    LUS_pc = np.vstack((cyl_pc, hem_pc))
    
    return LUS_pc


def create_Lap_pc(radius, grid_size):
    """"
    Doc:
    Creates laparoscope point cloud as a siple cylinder
    Could be developed further if more complex model required
    """
    
    cyl_pc = createPcCylinder(length=30,
                              radius=radius,
                              grid_size=grid_size)
    
    return cyl_pc
    

def transform_pc(pc, pose):
    """"
    Doc:
    Rigid transformation of a point cloud
    """
    translation = pose[0:3].reshape(1, 3)[0]
    direction_vec = pose[3:6].reshape(1, 3)[0]

    np.random.seed(26)
    z_vec = direction_vec
    y_vec = np.random.randn(3)
    y_vec -= np.dot(y_vec, z_vec)*z_vec
    x_vec = np.cross(y_vec, z_vec)
    
    x_vec /= np.linalg.norm(x_vec)
    y_vec /= np.linalg.norm(y_vec)
    z_vec /= np.linalg.norm(z_vec)
    
    i_vec = np.array([1, 0, 0])
    j_vec = np.array([0, 1, 0])
    k_vec = np.array([0, 0, 1])
    
    R = np.array([
      [np.dot(x_vec, i_vec), np.dot(y_vec, i_vec), np.dot(z_vec, i_vec)],
      [np.dot(x_vec, j_vec), np.dot(y_vec, j_vec), np.dot(z_vec, j_vec)],
      [np.dot(x_vec, k_vec), np.dot(y_vec, k_vec), np.dot(z_vec, k_vec)]
    ])

    T = np.block(
        [R, translation.reshape(3, 1)],
    )

    pc_T = (T@(np.hstack([pc, np.ones((np.shape(pc)[0], 1))])).T).T

    return pc_T
   


def createPlanePc(tl_pt, R, grid_size):
    """"
    Doc:
    Creates a finite plane point cloud
    """
    x = np.arange(0, 60, grid_size)
    y = np.arange(0, 100, grid_size)
    xv, yv = np.meshgrid(x, y)

    pc = np.dstack([xv, yv]).reshape(-1, 2)
    pc = np.hstack([pc, np.ones((np.shape(pc)[0], 1))])

    pc = (R@pc.T).T

    pc = pc + tl_pt

    return pc
 
################################ LUS primites functions ################################

def rotFromAx(u, a):
    """
    Doc:
    Generates a rotation matrix for a rotation around a given axis
    Exponent could be used instead
    """
    u = u/np.linalg.norm(u)
    ux, uy, uz = u

    R00 = math.cos(a) + (ux**2)*(1 - math.cos(a))
    R01 = ux*uy*(1 - math.cos(a)) - uz*math.sin(a)
    R02 = ux*uz*(1 - math.cos(a)) + uy*math.sin(a)
    R10 = uy*ux*(1 - math.cos(a)) + uz*math.sin(a)
    R11 = math.cos(a) + (uy**2)*(1 - math.cos(a))
    R12 = uy*uz*(1 - math.cos(a)) - ux*math.sin(a)
    R20 = uz*ux*(1 - math.cos(a)) - uy*math.sin(a)
    R21 = uz*uy*(1 - math.cos(a)) + ux*math.sin(a)
    R22 = math.cos(a) + (uz**2)*(1 - math.cos(a))

    R = np.array([
          [R00, R01, R02],
          [R10, R11, R12],
          [R20, R21, R22]
    ])

    return R


def Plucker2Cartesian(u, m, length, offset):
    """
    Doc:
    Creates a finite line from Plucker params
    """
    u = u/np.linalg.norm(u)
    m_dir = m/np.linalg.norm(m)
    m_mag = np.linalg.norm(m)
    assert abs(np.dot(u, m_dir)) < 1e-10, "u must be perpendicular to m"

    p_perp = np.cross(u, m)  # closest point on u line to the origin

    pt1 = p_perp + 0.5*length*u + offset*u
    pt2 = p_perp - 0.5*length*u + offset*u

    return pt1, pt2


def lus_primitives_from_pose(sc, u, radius, camera_matrix):
    """
    Doc:
    Generates LUS primites given its pose
    """
    u = u / np.linalg.norm(u)

    pt1 = sc
    pt2 = pt1 + 60*u
    
    m_dir = np.cross(pt1, pt2)
    m_dir = m_dir / np.linalg.norm(m_dir)
    
    m_mag = np.linalg.norm(pt1)*math.sin(math.acos(np.dot(pt1, u)/np.linalg.norm(pt1)))
    
    m = m_mag*m_dir
    
    alpha = math.asin(radius/m_mag)
    
    m1_dir = rotFromAx(u=u, a=alpha)@(m_dir.reshape(3, 1))
    m1_dir = m1_dir.reshape(1, 3)[0]
    m1_mag = ((m_mag**2 - radius**2)**0.5)
    
    m2_dir = rotFromAx(u=u, a=-alpha)@(m_dir.reshape(3, 1))
    m2_dir = m2_dir.reshape(1, 3)[0]
    m2_mag = ((m_mag**2 - radius**2)**0.5)
    
    m1 = m1_mag*m1_dir
    m2 = m2_mag*m2_dir


    L1_pt1_lap, L1_pt2_lap = Plucker2Cartesian(u=u, m=m1, length=300, offset=120)
    L2_pt1_lap, L2_pt2_lap = Plucker2Cartesian(u=u, m=m2, length=300, offset=120)

    
    L1_pt1_px = (camera_matrix@L1_pt1_lap.reshape(3, 1)).reshape(1, 3)[0]
    L1_pt1_px = L1_pt1_px / L1_pt1_lap[-1]
    L1_pt1_px = L1_pt1_px[0:2]
    
    L1_pt2_px = (camera_matrix@L1_pt2_lap.reshape(3, 1)).reshape(1, 3)[0]
    L1_pt2_px = L1_pt2_px / L1_pt2_lap[-1]
    L1_pt2_px = L1_pt2_px[0:2]
    
    A1 = L1_pt2_px[1] - L1_pt1_px[1]
    B1 = L1_pt1_px[0] - L1_pt2_px[0]
    C1 = (L1_pt2_px[0]*L1_pt1_px[1]) - (L1_pt1_px[0]*L1_pt2_px[1])
    L1_px_eq = np.array([A1, B1, C1])
    L1_px_eq = L1_px_eq/np.linalg.norm(L1_px_eq[0:2])
    # L1_px = np.linspace(L1_pt1_px, L1_pt2_px, 1000)


    L2_pt1_px = (camera_matrix@L2_pt1_lap.reshape(3, 1)).reshape(1, 3)[0]
    L2_pt1_px = L2_pt1_px / L2_pt1_lap[-1]
    L2_pt1_px = L2_pt1_px[0:2]
    
    L2_pt2_px = (camera_matrix@L2_pt2_lap.reshape(3, 1)).reshape(1, 3)[0]
    L2_pt2_px = L2_pt2_px / L2_pt2_lap[-1]
    L2_pt2_px = L2_pt2_px[0:2]
    
    A2 = L2_pt2_px[1] - L2_pt1_px[1]
    B2 = L2_pt1_px[0] - L2_pt2_px[0]
    C2 = (L2_pt2_px[0]*L2_pt1_px[1]) - (L2_pt1_px[0]*L2_pt2_px[1])
    L2_px_eq = np.array([A2, B2, C2])
    L2_px_eq = L2_px_eq/np.linalg.norm(L2_px_eq[0:2])
    # L2_px = np.linspace(L2_pt1_px, L2_pt2_px, 1000)

    n_pt_sphere = 100

    x0, y0, z0 = sc
    r = radius
    z1 = 1
    
    x, y = symbols('x y')
    eq = (x*x0 + y*y0 + z1*z0)**2 - (x**2 + y**2 + z1**2)*(x0**2 + y0**2 + z0**2 - r**2)
    p, q = solve(eq, [x, y])
    
    eq_domain = ((-r**2 + x0**2 + y0**2 + z0**2)*(r**2*y**2 + r**2*z1**2 -
                                                  y**2*z0**2 + 2*y*y0*z0*z1 -
                                                  y0**2*z1**2))
    domain_start, domain_end = solve(eq_domain, [y])
    domain_start = float(domain_start)
    domain_end = float(domain_end)
    domain_length = abs(domain_end - domain_start)
    domain_mid = (domain_start + domain_end)/2
    
    ellps_y = np.linspace(start=domain_start, stop=domain_end, num=n_pt_sphere + 1,
                        endpoint=False, retstep=False, dtype=None, axis=0)[1:]  # Exclude init pt
    
    # ellps_y = np.random.choice(ellps_y, n_pt_sphere)  # Random from tip pts
    
    ellps_x = np.empty(len(ellps_y))
    for i, y_val in enumerate(ellps_y):
        ellps_x[i] = p[0].subs(y, y_val)
    
    ellps_pts = np.hstack((ellps_x.reshape(n_pt_sphere, 1), ellps_y.reshape(n_pt_sphere, 1), z1*np.ones((n_pt_sphere, 1))))
    
    
    ellps_px = (camera_matrix@ellps_pts.T).T
    ellps_px = ellps_px[:, 0:2]

    # return L1_px_eq, L2_px_eq, ellps_px, L1_px, L2_px
    return L1_px_eq, L2_px_eq, ellps_px
        


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import time
import psutil
import h5py

import numpy as np
import pandas as pd


from scipy.interpolate import interp1d
import scipy.integrate as integrate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UTILITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def func_stats(method):
    def stats(*args, **kw):
        p = psutil.Process()
        
        pmem0 = p.memory_info()
        t0 = time.time()
        
        result = method(*args, **kw)
        
        t1 = time.time()
        pmem1 = p.memory_info()
        
        print('\n','~'*5, 'Runtime stats for', method.__name__, '~'*5)
        print('Time elapsed:', t1-t0, 'sec')
        print('MEM before:', pmem0.rss/1e9, 'GB (Physical); ', 
              pmem0.vms/1e9, 'GB (Virtual)')
        print('MEM after: ', pmem1.rss/1e9, 'GB (Physical); ', 
              pmem1.vms/1e9, 'GB (Virtual)')
        
        return result
    
    return stats

def get_proc():
    mem = psutil.virtual_memory()
    p = max(1, int(100./mem.percent)-1)
    p = min(p, psutil.cpu_count())
    print('n_proc:', p)
    return p

def pd_from_hdf(filename, fields):
    with h5py.File(filename,'r') as f:
        x = f[fields[0]][()]

        data = pd.DataFrame(0, index=np.arange(len(x)), columns=fields, dtype='<f8')
        data[fields[0]] = x
        
        for i in range(1,len(fields)):
            x = f[fields[i]][()]
            data[fields[i]] = x
    
    return data

def load_raw(filename):
    print('Loading data from: ' + filename)
    if filename[-4:] == '.csv':
        # this throws a warning because of a bug in numpy
        # https://stackoverflow.com/questions/48818335/why-pandas-read-csv-issues-this-warning-elementwise-comparison-failed?rq=1
        return pd.read_csv(filename, index_col=0) 
    elif filename[-4:] == '.npy':
        return pd.DataFrame(np.load(filename))
    else:
        print('Invalid file type: ' + filename[-4:])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COSMOLOGY FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
d_from_z_sci, z_from_d_sci, d_clu, vH_clu = None,None,None,None

def H_from_z(z, cosmo):
    # H(z): Hubble parameter from redshift
    # H = cosmo['h'] * (100.) * (cosmo['Omega_m'] * (1 + z)**3 + cosmo['Omega_l'])**0.5
    H = (100.) * (cosmo['Omega_m'] * (1 + z)**3 + cosmo['Omega_l'])**0.5
    return H # h*(km/s)/Mpc


def d_from_z(z, cosmo):
    # d(z): Comoving distance from redshift
    d = integrate.quad(lambda x: cosmo['c'] / H_from_z(x, cosmo), 0, z)
    return d[0] # Mpc/h




def vH_from_z(z, cosmo):
    # v_H(z): Hubble recession velocity from redshift
    vH = cosmo['c'] * ((1 + z)**2 - 1) / ((1 + z)**2 + 1)
    return vH


def add_v(v1, v2, cosmo):
    # Add velocities relativistically
    v_sum = (v1 + v2) / (1 + v1 * v2 / cosmo['c']**2)
    return v_sum


def sub_v(v1, v2, cosmo):
    # Subtract velocities relativistically
    v_diff = (v1 - v2) / (1 - v1 * v2 / cosmo['c']**2)
    return v_diff





def cal_v_rel(host_pos, host_v, gal_pos, gal_v, cosmo):
    """
    Given host positions, velocities and galaxy position, velocities,
    calculate relative velocities. See Hy's notes.
    """
    global z_from_d_sci, d_clu, vH_clu
    
    dist_gal_clu = gal_pos - host_pos # Mpc/h

    d_gal = d_clu + dist_gal_clu # Mpc/h + Mpc/h

    # next two lines account for if d_gal is negative
    z_gal = z_from_d_sci(np.abs(d_gal)) # Mpc/h -> z

    vH_gal = vH_from_z(z_gal, cosmo)*np.sign(d_gal)

    v_gal = add_v(vH_gal, gal_v, cosmo)

    v_clu = add_v(vH_clu, host_v, cosmo)

    v_rel = sub_v(v_gal, v_clu, cosmo)

    return v_rel


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ROTATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def spherical_to_cartesian(R, theta, phi):
    return (R * np.sin(theta) * np.cos(phi), R * np.sin(theta) * np.sin(phi),
            R * np.cos(theta))

def rot_matrix_LOS(theta, phi):
    """
    Generate a rotation matrix from (theta,phi) as defined on the unit
    sphere. Transforms the LOS z-axis to the new (theta,phi)
    coordinates. I did my best to simplify the math. See
    https://en.wikipedia.org/wiki/Rotation_matrix
    """

    if (theta == 0) & (phi == 0):
        return np.identity(3)

    old_pos = spherical_to_cartesian(1, theta, phi)
    new_pos = (0, 0, 1)

    # rotation axis
    u = np.cross(old_pos, new_pos).astype('<f4')
    u /= np.linalg.norm(u)

    # rotation angle
    th = np.arccos(np.dot(old_pos, new_pos))

    u_cross = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

    # rotation matrix
    R = np.cos(th) * np.identity(3) + \
        np.sin(th) * u_cross + \
        (1 - np.cos(th)) * np.tensordot(u, u, axes=0)

    return R


def fibonacci_sphere(N=1, randomize=True):
    """
    Generate a set of N angles, (theta, phi), 'evenly' distributed
    on the unit sphere. In truth, it is Fibonacci-sphere distributed.
    See https://stackoverflow.com/questions/9600801/evenly-distributing-
    n-points-on-a-sphere
    """
    if N < 1:
        return []

    rnd = np.random.rand() * N if randomize else 1.

    points = []

    dz = 2. / N
    dphi = np.pi * (3. - np.sqrt(5))  # Golden angle

    for i in range(N):
        z = ((i * dz) - 1) + (dz / 2.)

        theta = np.arccos(z)
        phi = ((i + rnd) % N) * dphi

        points.append((theta, phi))

    if randomize:
        np.random.shuffle(points)

    return points

def match_angles(new_angs, template_angs):
    new_angs = np.array(new_angs)
    template_angs = np.array(template_angs)
    
    new_vecs = np.array(spherical_to_cartesian(1, *(new_angs.T)))
    template_vecs = np.array(spherical_to_cartesian(1, *(template_angs.T)))
    
    xang = np.abs(np.arccos(np.matmul(new_vecs.T, template_vecs)))
    
    out = []
    while len(xang) > 0:
        i_min = np.unravel_index(np.argmin(xang, axis=None), xang.shape)
        
        out.append(template_angs[i_min[1]])
        xang = np.delete(xang, i_min[0], axis=0)
        xang = np.delete(xang, i_min[1], axis=1)
        template_angs = np.delete(template_angs, i_min[1], axis=0)
        
    return out

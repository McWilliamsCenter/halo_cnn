"""
This script is used to generate a mock catalog from MDPL2 Rockstar data.

The main procedures of this script are:
    1. Load and organize all raw data and mock observation parameters.
       Seperate the data into 'host' information and 'galaxy'
       information.

    2. Iterate through all galaxies and assign them to their hosts.
       Galaxies which are gravitationally bound to their hosts are
       designated as the 'true members' of that host.

    3. Pad the galaxy data by creating copies of the galaxies which are
       near edges of the box. Keep track of the galaxies and their
       respective copies.

    4. Reassign true members to hosts which are close to the edges of
       the box. The padded galaxies might be closer to the host center
       than the originals.

    5. Cut cylinders
        a. Rotate the box randomly. All x,y,z and vx,vy,vz data of hosts
           and galaxies are transformed by a rotation matrix R. The
           following steps will run in parallel for each rotated box.
        b. Initialize a kd-tree along the (rotated) x,y positions of
           galaxies.
        c. Iterate through all hosts and identify all galaxies within
           [aperature] of the host center along x,y positions.
        d. For each of these galaxies, calculate relative LOS velocity
           to the cluster.
        e. Cut off relative velocity of the members by v_cut. The
           galaxies which fall within the velocity cut are part of the
           contaminated catalog.
        f. Calculate relative LOS velocity for all true members of the
           host. These galaxies are part of the pure catalog.

    6. Organize output and save data

"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
import pandas as pd
import time

import scipy.integrate as integrate
import multiprocessing as mp
from scipy.spatial import KDTree
from collections import OrderedDict, defaultdict

from tools.catalog import Catalog


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
par = OrderedDict([
    ('wdir',   '/home/mho1/scratch/halo_cnn'),
    ('in_folder',   'data_raw'),
    ('out_folder',   'data_mocks'),

    # UM with z=0.117
    ('catalog_name',    'Rockstar_UM_z=0.117'),
    ('host_file',       'cosmosim/MDPL2_Rockstar_snap:120_v3.csv'),
    ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.895100.npy'),
    ('z',               0.117),

    ('host_min_mass',   10**(13.5)),
    ('gal_min_mass',    10**(9.5)),
    ('gal_mass_def',    'obs_sm'),
    ('min_richness',    10),

    ('cut_size',        'medium'),
    ('true_members',    'halo'),  # vir or halo

    ('dn_dlogm',        10.**-5),
    ('dlogm',           0.01),
    ('boost_minmass',   10**14),
    ('min_rotations',   3),

    ('cosmo',   {'H_0': 100,  # [km/s/(Mpc/h)]
                 'Omega_m': 0.307115,
                 'Omega_l': 0.692885,
                 'c': 299792.458  # [km/s]
                 })
])
# For running
n_proc = 4
np.random.seed(9834234)

# For debugging
debug = False

if debug:
    print('\n' + 20 * '>' + ' DEBUGGING ' + 20 * '<')
    par['min_richness'] = 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETER HANDLING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\n~~~~~ GENERATING ' + par['catalog_name'] + ' ~~~~~~')

for key in par.keys():
    print(key + ' : ' + str(par[key]))

cosmo = par['cosmo']

if par['cut_size'] == 'small':
    aperture = 1.1  # the radial aperture in comoving Mpc/h
    vcut = 1570.  # maximum velocity is vmean +/- vcut in km/s
elif par['cut_size'] == 'medium':
    aperture = 1.6  # Mpc/h
    vcut = 2200.  # km/s
elif par['cut_size'] == 'large':
    aperture = 2.3  # Mpc/h
    vcut = 3785.  # km/s
else:
    raise Exception('Invalid cut_size')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UTILITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_raw(filename):
    print('Loading data from: ' + filename)
    if filename[-4:] == '.csv':
        return pd.read_csv(filename, index_col=0)
    elif filename[-4:] == '.npy':
        return pd.DataFrame(np.load(filename))
    else:
        print('Invalid file type: ' + filename[-4:])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COSMOLOGY FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def H_from_z(z, cosmo):
    # H(z): Hubble parameter from redshift
    H = cosmo['H_0'] * (cosmo['Omega_m'] * (1 + z)**3 + cosmo['Omega_l'])**0.5
    return H


def d_from_z(z, cosmo):
    # d(z): Comoving distance from redshift
    d = integrate.quad(lambda x: cosmo['c'] / H_from_z(x, cosmo), 0, z)
    return d[0]


"""
Since d(z) has no simple, analytic form, but is smooth and continuous,
we will sample it at many z and use numpy's interpolation function to
reduce calculation time.
"""

z_samp = np.arange(0, 5, 0.0001)
d_samp = [d_from_z(i, cosmo) for i in z_samp]


def d_from_z_nump(z):
    # d(z) interpolation: Comoving distance from redshift
    d_nump = np.interp(z, z_samp, d_samp)
    return d_nump


def z_from_d_nump(d):
    # z(d) interpolation: Redshift from comoving distance
    z_nump = np.interp(d, d_samp, z_samp)
    return z_nump


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


"""
We'll assume that each host (cluster) is centered at z=par['z'].
Therefore, comoving distances and Hubble recession velocities of the
hosts are constant.
"""
d_clu = d_from_z(par['z'], cosmo)
vH_clu = vH_from_z(par['z'], cosmo)


def cal_v_rel(host_pos, host_v, gal_pos, gal_v, cosmo):
    """
    Given host positions, velocities and galaxy position, velocities,
    calculate relative velocities. See Hy's notes.
    """

    dist_gal_clu = gal_pos - host_pos

    d_gal = d_clu + dist_gal_clu

    z_gal = z_from_d_nump(d_gal)

    vH_gal = vH_from_z(z_gal, cosmo)

    v_gal = add_v(vH_gal, gal_v, cosmo)

    v_clu = add_v(vH_clu, host_v, cosmo)

    v_rel = sub_v(v_gal, v_clu, cosmo)

    return v_rel


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ROTATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def rot_matrix_LOS(theta, phi):
    """
    Generate a rotation matrix from (theta,phi) as defined on the unit
    sphere. Transforms the LOS z-axis to the new (theta,phi)
    coordinates. I did my best to simplify the math. See
    https://en.wikipedia.org/wiki/Rotation_matrix
    """

    if (theta == 0) & (phi == 0):
        return np.identity(3)

    old_pos = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
               np.cos(theta))
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


t0 = time.time()


# ~~~~~~~~~~~~~~~~~~~~~~~~ LOADING, PREPROCESSING DATA ~~~~~~~~~~~~~~~~~~~~~~~~
print('\nLoading data...')
host_data = load_raw(os.path.join(par['wdir'], par['in_folder'],
                                  par['host_file']))

if par['gal_file'] is None:
    gal_data = host_data
else:
    gal_data = load_raw(os.path.join(par['wdir'], par['in_folder'],
                                     par['gal_file']))
    gal_data = gal_data.rename(columns={'upid': 'upId', 'mpeak': 'Macc',
                                        'mvir': 'Mvir'})


# Cluster, galaxy constraints
print('\nFinding hosts, galaxies...')
host_data = host_data[(host_data['upId'] == -1) &
                      (host_data['M200c'] >= par['host_min_mass'])]

gal_data = gal_data[(gal_data['upId'] != -1) &
                    (gal_data[par['gal_mass_def']] >= par['gal_min_mass'])]

if debug:
    host_data = host_data.sample(int(10**3.5), replace=False)
    gal_data = gal_data.sample(int(10**5.5), replace=False)

host_data = host_data.reset_index(drop=True)
gal_data = gal_data.reset_index(drop=True)

print('host_data length: ' + str(len(host_data)))
print('gal_data length: ' + str(len(gal_data)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIND TRUE MEMBERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\nAssigning true members...')

# Relates host rockstarid to indices of member galaxies
host_members = defaultdict(lambda: [])

for i in range(len(gal_data)):
    if i % (10**5) == 0:
        print(str(int(i / (10**5))) + ' / ' +
              str(int(len(gal_data) / (10**5))))
    # Assign to host

    host_members[gal_data.loc[i, 'upId']].append(i)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PADDING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
to_pad is a dictionary which relates tuples to an array of galaxy
indices. The tuples have length 3 and correspond to the x,y,z axes,
respectively. For a given axis n, if the value within the tuple is
'True', the galaxies within the related array will have their 'n'-axis
added one box_length when padding. If the tuple value is 'False', the
galaxies within the related array will have their 'n'-axis subtracted
one box_length when padding. If 'None' appears, the 'n'-axis is left
alone while padding. I did this to automate the padding process and
make it O(n).
"""

print('\nDetermining padding regions...')
max_gal_v = np.sqrt((gal_data['vx']**2 + gal_data['vy']**2 +
                     gal_data['vz']**2).max())
max_clu_v = np.sqrt((host_data['vx']**2 + host_data['vy']**2 +
                     host_data['vz']**2).max())

# max possible distance in the non-relativistic case
pad_size = 1.0 * (vcut + max_gal_v + max_clu_v) / H_from_z(par['z'], cosmo)

box_length = 1000

print('pad_size/box_length: ' + str(pad_size / box_length))

pad_regions = []
pad_directions = (None, False, True)

for i in pad_directions:
    for j in pad_directions:
        for k in pad_directions:
            pad_regions.append((i, j, k))

_ = pad_regions.remove((None, None, None))

axes = ('x', 'y', 'z')


def in_pad_region(key):
    # Assign to padding regions. To be used in multiprocessing
    out = np.ones(shape=(len(gal_data), len(axes)))

    for j in range(len(key)):
        if key[j] is True:
            out[:, j] = (gal_data[axes[j]] < pad_size)
        elif key[j] is False:
            out[:, j] = (gal_data[axes[j]] > (box_length - pad_size))

    return np.argwhere(np.sum(out, axis=1) == len(axes)).flatten()


with mp.Pool(processes=n_proc) as pool:
    # each process checks a different padding region
    to_pad_ind = pool.map(in_pad_region, pad_regions, chunksize=1)

print('Padding regions: ')
for i in range(len(pad_regions)):
    s = ''
    for j in range(len(pad_regions[i])):
        if pad_regions[i][j] is True:
            s += axes[j] + '+ '
        elif pad_regions[i][j] is False:
            s += axes[j] + '- '

    print(s + ' : ' + str(len(to_pad_ind[i])))

num_padded = np.sum([len(to_pad_ind[i]) for i in range(len(to_pad_ind))])

c = len(gal_data)

gal_data = gal_data.append(pd.DataFrame(
    np.zeros(shape=(num_padded, gal_data.shape[1])),
    columns=gal_data.columns), ignore_index=True)

# Relates indices of original galaxies to their padded copy indices
pad_gal_copies = defaultdict(lambda: [])

print('\nPadding...')
for i in range(len(pad_regions)):
    print(str(i + 1) + ' / ' + str(len(pad_regions)))

    c_end = c + len(to_pad_ind[i])
    gal_data.values[c: c_end, :] = gal_data.iloc[to_pad_ind[i]].values

    for j in range(len(to_pad_ind[i])):
        pad_gal_copies[to_pad_ind[i][j]].append(c + j)

    for j in range(len(pad_regions[i])):
        if pad_regions[i][j] is True:
            gal_data.loc[c: c_end - 1, axes[j]] += box_length
        elif pad_regions[i][j] is False:
            gal_data.loc[c: c_end - 1, axes[j]] -= box_length

    c = c_end

print('Padded gal_data length: ' + str(len(gal_data)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ REASSIGNING TRUE MEMBERS ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Host-galaxy relationships might take place over a periodic boundary.
Therefore, we must check if any padded copies of galaxies are closer to
their respective hosts than the original galaxies
"""


def reassign_true(host_i):
    host_pos = host_data.iloc[host_i][['x', 'y', 'z']]
    host_mem = host_members[host_data.at[host_i, 'rockstarId']]

    for j in range(len(host_mem)):
        true_memb_dist = np.sqrt(np.sum(
            (gal_data.iloc[host_mem[j]][['x', 'y', 'z']] - host_pos)**2))
        if (true_memb_dist > host_data.at[host_i, 'Rvir'] / 1000.) & \
           (len(pad_gal_copies[host_mem[j]]) > 0):
            close_gal = np.sqrt(np.sum(
                (gal_data.iloc[pad_gal_copies[host_mem[j]]][['x', 'y', 'z']] -
                 host_pos)**2, axis=1)).idxmin()

            if np.sqrt(
                np.sum((gal_data.iloc[close_gal][['x', 'y', 'z']]
                        - host_pos)**2)) < true_memb_dist:
                host_mem[j] = close_gal

    return host_mem


print('\nReassigning true members to account for padding...')
with mp.Pool(processes=n_proc) as pool:
    # each process checks a different padding region
    reassignments = pool.map(reassign_true, range(len(host_data)))

for i in range(len(host_data)):
    if i % (10**4) == 0:
        print(str(int(i / (10**4))) + ' / ' +
              str(int(len(host_data) / (10**4))))
    host_members[host_data.at[i, 'rockstarId']] = reassignments[i]


print('\nReducing dataset...')  # Dropping unnecessary data fields

host_drop = ['row_id', 'upId', 'pId', 'descId', 'breadthFirstId', 'scale']
gal_drop = []

host_data = host_data[[i for i in host_data.columns.values if i not in
                       host_drop]]
gal_data = gal_data[[i for i in gal_data.columns.values if i not in gal_drop]]


# ~~~~~~~~~~~~~~~~~~~~ CALCULATING BOOSTED ROTATION ANGLES ~~~~~~~~~~~~~~~~~~~~

print('Calculating boosted rotation angles...')

host_rots = pd.DataFrame(0, index=host_data.index,
                         columns=['M200c', 'num_rot'])

host_rots.loc[:, 'M200c'] = np.log10(host_data['M200c'])
host_rots.loc[:, 'num_rot'] = 0

host_rots = host_rots.sort_values(by='M200c')

start = np.argwhere(host_rots['M200c'] > np.log10(par['boost_minmass'])).min()

chunks = np.linspace(start, len(host_rots), n_proc + 1)
chunks = [(int(chunks[i]), int(chunks[i + 1])) for i in range(len(chunks) - 1)]


def num_rot(chunk):
    out = [0] * (chunk[1] - chunk[0])
    for i in range(chunk[0], chunk[1]):
        window = [0, 1]
        while (host_rots['M200c'].iloc[window[0]] < host_rots['M200c'].iloc[i]
                - par['dlogm'] / 2.):
            window[0] += 1
        while (window[1] < len(host_rots)):
            if(host_rots['M200c'].iloc[window[1]] >= host_rots['M200c'].iloc[i]
                    + par['dlogm'] / 2.):
                break
            window[1] += 1

        i_dn_dlogm = (window[1] - window[0]) / 1000**3 / par['dlogm']

        out[i - chunk[0]] = int(par['dn_dlogm'] / i_dn_dlogm) + 1  # fix

    return out


with mp.Pool(processes=n_proc) as pool:
    # each process checks a different padding region
    chunk_out = pool.map(num_rot, chunks)

print('Setting chunks')
for i in range(len(chunk_out)):
    print(i, '/', len(chunk_out))
    host_rots.iloc[chunks[i][0]:chunks[i][1],
                   host_rots.columns.get_loc('num_rot')] = chunk_out[i]

host_rots.iloc[np.argwhere(host_rots['num_rot'] < par['min_rotations']),
               host_rots.columns.get_loc('num_rot')] = par['min_rotations']

rot_assign = pd.DataFrame(
    True, index=host_rots.index,
    columns=[(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)])

for n_rot in host_rots['num_rot'].unique():
    ang_list = fibonacci_sphere(n_rot - par['min_rotations'])

    for ang in ang_list:
        rot_assign[ang] = host_rots['num_rot'] == n_rot

print('# Rotations:' + str(host_rots['num_rot'].unique()))


# ~~~~~~~~~~~~~~~~~~~ CALCULATING ROTATED MOCK OBSERVATIONS ~~~~~~~~~~~~~~~~~~~


def cut_mock(angle=None):
    """
    Given a set of rotation angles, rotate catalog. Iterate through clusters.
    Cut a long cylinder through the entire sim around projected cluster center
    (called pillars). Calculate all vlos. Create pure, contaminated catalogs.

    This is wrapped in a function so it can be parallelized.
    """

    # ~~~~~ rotate galaxy-host positions+velocities ~~~~~
    R = rot_matrix_LOS(*angle)

    gal_rot = pd.DataFrame(
        np.concatenate((np.matmul(gal_data[['x', 'y', 'z']].values, R.T),
                        np.matmul(gal_data[['vx', 'vy', 'vz']].values, R.T)),
                       axis=1), columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    host_rot = pd.DataFrame(
        np.concatenate((np.matmul(host_data[['x', 'y', 'z']].values, R.T),
                        np.matmul(host_data[['vx', 'vy', 'vz']].values, R.T)),
                       axis=1), columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])

    # ~~~~~ initialize projected and 3D KDTrees ~~~~~
    gal_tree_proj = KDTree(gal_rot[['x', 'y']], leafsize=50)
    gal_tree_full = KDTree(gal_rot[['x', 'y', 'z']], leafsize=50)

    # ~~~~~ initialize catalogs ~~~~~
    # empty catalog files
    pure_catalog = Catalog(prop=host_data, gal=[])
    contam_catalog = Catalog(prop=host_data, gal=[])

    # record which clusters were successfully measured
    pure_ind = []
    contam_ind = []

    # output member galaxy information
    gal_prop = [('Mvir', '<f8'), ('Macc', '<f8')]
    if par['gal_file'] is not None:
        gal_prop += [('obs_sm', '<f8'), ('obs_sfr', '<f8')]

    dtype_gal = [('xproj', '<f8'), ('yproj', '<f8'), ('zproj', '<f8'),
                 ('Rproj', '<f8'), ('vlos', '<f8'),
                 ('memb_halo', '?'), ('memb_vir', '?'),
                 *gal_prop
                 ]

    # ~~~ iterate through hosts, assign member galaxies, calculate observables

    for i in rot_assign.index.values[rot_assign[angle]]:  # for each cluster

        # Grab galaxy members as assigned by halo algo
        mem_halo_ind = host_members[host_data.loc[i, 'rockstarId']]

        # Sample KDTree to find potential projected and virial members
        mem_proj_ind = gal_tree_proj.query_ball_point(
            host_rot.loc[i, ['x', 'y']], aperture)
        mem_vir_ind = gal_tree_full.query_ball_point(
            host_rot.loc[i, ['x', 'y', 'z']], host_data.loc[i, 'Rvir'] / 1000)

        # calculate relative velocities of all potential projected members
        v_rel_proj = cal_v_rel(host_rot.loc[i, 'z'],
                               host_rot.loc[i, 'vz'],
                               gal_rot.loc[mem_proj_ind, 'z'],
                               gal_rot.loc[mem_proj_ind, 'vz'],
                               cosmo)

        # restrict projected member list to those with |v_los|<v_cut
        v_rel_proj = v_rel_proj[np.abs(v_rel_proj) < vcut]
        mem_proj_ind = v_rel_proj.index.to_series()

        # identify true members for pure catalog
        if par['true_members'] == 'halo':
            mem_true_ind = mem_halo_ind
        elif par['true_members'] == 'vir':
            mem_true_ind = mem_vir_ind
        else:
            raise Exception('Unknown true member assignment scheme: %s'
                            % par['true_members'])

        # contaminated catalog
        if len(mem_proj_ind) > par['min_richness']:

            # initialize array of output member galaxy information
            clu_gals = np.zeros(shape=(len(mem_proj_ind)),
                                dtype=dtype_gal)

            # assign relevant information
            clu_gals[['xproj', 'yproj', 'zproj']] = (
                gal_rot.loc[mem_proj_ind, ['x', 'y', 'z']]
                - host_rot.loc[i, ['x', 'y', 'z']]).to_records(index=False)
            clu_gals['Rproj'] = np.sqrt(
                clu_gals['xproj']**2 + clu_gals['yproj']**2)

            clu_gals['vlos'] = v_rel_proj

            clu_gals['memb_halo'] = [x in mem_halo_ind for x in mem_proj_ind]
            clu_gals['memb_vir'] = [x in mem_vir_ind for x in mem_proj_ind]
            clu_gals[[p[0] for p in gal_prop]] = gal_data.loc[
                mem_proj_ind, [p[0] for p in gal_prop]].to_records(index=False)

            # append output galaxy information to catalog
            contam_catalog.gal.append(clu_gals)
            contam_ind.append(i)

        # pure catalog
        if len(mem_true_ind) > par['min_richness']:

            # calculate v_los for true members which may not fall in projected
            # cylinder
            mem_truenotproj_ind = [
                j for j in mem_true_ind if j not in mem_proj_ind]

            v_rel_truenotproj = cal_v_rel(host_rot.loc[i, 'z'],
                                          host_rot.loc[i, 'vz'],
                                          gal_rot.loc[
                mem_truenotproj_ind, 'z'],
                gal_rot.loc[
                mem_truenotproj_ind, 'vz'],
                cosmo)

            v_rel_true = v_rel_proj.append(v_rel_truenotproj)[mem_true_ind]

            # initialize array of output member galaxy information
            clu_gals = np.zeros(shape=(len(mem_true_ind)),
                                dtype=dtype_gal)

            # assign relevant information
            clu_gals[['xproj', 'yproj', 'zproj']] = (
                gal_rot.loc[mem_true_ind, ['x', 'y', 'z']]
                - host_rot.loc[i, ['x', 'y', 'z']]).to_records(index=False)
            clu_gals['Rproj'] = np.sqrt(
                clu_gals['xproj']**2 + clu_gals['yproj']**2)

            clu_gals['vlos'] = v_rel_true

            clu_gals[[p[0] for p in gal_prop]] = gal_data.loc[
                mem_true_ind, [p[0] for p in gal_prop]].to_records(index=False)

            if par['true_members'] == 'halo':
                clu_gals['memb_halo'] = True
                clu_gals['memb_vir'] = [x in mem_vir_ind for x in mem_true_ind]
            elif par['true_members'] == 'vir':
                clu_gals['memb_halo'] = [x in mem_halo_ind
                                         for x in mem_true_ind]
                clu_gals['memb_vir'] = True

            # append output galaxy information to catalog
            pure_catalog.gal.append(clu_gals)
            pure_ind.append(i)

    # restrict output catalogs to only correctly calulated clusters (e.g.
    # above min_richness)
    pure_catalog.prop = pure_catalog.prop.iloc[pure_ind]
    contam_catalog.prop = contam_catalog.prop.iloc[contam_ind]

    # delete unnecessary data
    del gal_rot
    del host_rot

    # output and return
    print('Finished ' + str(angle))

    return (pure_catalog, contam_catalog)


print('\nCutting cylinders...')

with mp.Pool(processes=n_proc) as pool:
    catalogs = pool.map(cut_mock, rot_assign.columns.values, chunksize=1)

print('Done.')

pure_len = 0
contam_len = 0

for i in range(len(catalogs)):
    pure_len += len(catalogs[i][0].prop)
    contam_len += len(catalogs[i][1].prop)

print('\nCombining rotation catalogs...')

# Initialize catalogs

cat_par = OrderedDict([
    ('catalog_name',   par['catalog_name']),

    ('z',   par['z']),

    ('min_mass',   par['host_min_mass']),
    ('min_richness',   par['min_richness']),

    ('aperture',   aperture),
    ('vcut',   vcut),

    ('cosmo',   cosmo)
])


contam_catalog = Catalog(
    par=cat_par.copy(),
    prop=pd.DataFrame(np.zeros(
        shape=(contam_len, catalogs[0][1].prop.shape[1] + 1)),
        columns=np.append(catalogs[0][1].prop.columns.values, 'rotation')),
    gal=[None] * contam_len)

cat_par['aperture'] = None
cat_par['vcut'] = None

pure_catalog = Catalog(
    par=cat_par.copy(),
    prop=pd.DataFrame(np.zeros(
        shape=(pure_len, catalogs[0][0].prop.shape[1] + 1)),
        columns=np.append(host_data.columns.values, 'rotation')),
    gal=[None] * pure_len)


pure_c = 0
contam_c = 0

for i in range(len(catalogs)):
    pure, contam = catalogs[i]

    pure_c_end = pure_c + len(pure.prop)
    contam_c_end = contam_c + len(contam.prop)

    pure_catalog.prop.iloc[pure_c:pure_c_end, :-1] = pure.prop.values
    pure_catalog.prop.loc[pure_c:pure_c_end, 'rotation'] = i
    pure_catalog.gal[pure_c:pure_c_end] = pure.gal

    contam_catalog.prop.iloc[contam_c:contam_c_end, :-1] = contam.prop.values
    contam_catalog.prop.loc[contam_c:contam_c_end, 'rotation'] = i
    contam_catalog.gal[contam_c:contam_c_end] = contam.gal

    pure_c = pure_c_end
    contam_c = contam_c_end


pure_catalog.gal = np.array(pure_catalog.gal)
contam_catalog.gal = np.array(contam_catalog.gal)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ STATS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Calculating additional statistics...')

# Ngal
pure_catalog.prop['Ngal'] = [len(x) for x in pure_catalog.gal]
contam_catalog.prop['Ngal'] = [len(x) for x in contam_catalog.gal]

# sigv
gapper = lambda x: np.sqrt(np.sum((x-np.mean(x))**2)/(len(x)-1))
pure_catalog.prop['sigv'] = [gapper(x['vlos']) for x in pure_catalog.gal]
contam_catalog.prop['sigv'] = [gapper(x['vlos']) for x in contam_catalog.gal]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAVE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if not debug:
    print('Saving...')
    pure_catalog.save(os.path.join(par['wdir'],
                                   par['out_folder'],
                                   par['catalog_name'] + '_pure.p'))
    contam_catalog.save(os.path.join(par['wdir'],
                                     par['out_folder'],
                                     par['catalog_name'] + '_contam.p'))


print('\nAll done!')
print('Runtime: ' + str((time.time() - t0) / 60.) + ' minutes')

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

print('Importing modules...')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time

import multiprocessing as mp
from scipy.spatial import KDTree
from collections import OrderedDict, defaultdict

from tools.catalog import Catalog

import mock_tools
from mock_tools import *

import tracemalloc



# For running
np.random.seed(44323)
n_proc = 9

# For cache-ing intermediate data products
save_cache = True
load_cache = True

if len(sys.argv)==2:
    load_ind = int(sys.argv[1])
else:
    load_ind = None
    
print('Load index:', load_ind)

# For debugging
debug = False
tree_pool = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@func_stats
def build_config():
    global config
    
    config = OrderedDict([
        ('wdir',   '/hildafs/home/mho1/scratch/halo_cnn'),
        ('in_folder',   'data_raw'),
        ('out_folder',   'data_mocks'),
        
#         # UM Macc cut with z=0.022->0.0231
#         ('catalog_name',    'Rockstar_UM_z=0.0.022-0231_Macc1e11-large-fix'),
#         ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#         ('gal_file',        None), # 'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#         ('z',               0.0231), # coma
        
# #         # UM SM cut with z=0.022->0.0231
#         ('catalog_name',    'Rockstar_UM_z=0.0.022-0231_SM9.5-large-fix'),
#         ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#         ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#         ('z',               0.0231), # coma
        
#         # Uchuu with z=0.022->0.0231 and Macc>=1e11
#         ('catalog_name',    'Uchuu_z=0.022-0.231_VmaxP2.0-large-oldcorrect'),
#         ('host_file',       'uchuu/Uchuu_snap:049_m200c1e11.csv'),
#         ('gal_file',        None),
#         ('z',               0.0231), # coma
        
        # Magneticum with z=0.066340191 and m>=1e10.5
        ('catalog_name',    'Magneticum_z=0.066_m1e11-large'),
        ('host_file',       'magneticum/magn_snap136_cluster_fix+Dhayaa.csv'),        
        ('gal_file',        'magneticum/magn_snap136_galaxies_fix.csv'),
        ('z',               0.066340191),
        
#         # IllustriusTNG with z=0. and M*>=9.5
#         ('catalog_name',    'Illustrius_z=0_M*9.5-large'),
#         ('host_file',       'illustrius/TNG300-1_DES_Data_Halos_z0.csv'),
#         ('gal_file',        'illustrius/TNG300-1_DES_Data_Galaxies_z0.csv'),
#         ('z',               0.0),

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ('host_min_mass',   10**13.5), # 14 # 13.5
        ('gal_min_mass',    10**9.5), # 10**10.5), #11.3112), # (10.5)),# 10**12),  #10**11), (9.5)), # 
        ('gal_mass_def',    'm'), # 'Macc' #'Vmax_Mpeak' #'M*'), # 'obs_sm') 
        ('min_richness',    10),

        ('cut_size',        'large'),

        ('volume',          352.**3), # 1000.**3), # (Mpc/h)^3  # 2000.**3 # # 302.6**3), 
        ('dn_dlogm',        10.**-5.), # -5.2
        ('dlogm',           0.01),
        ('boost_minmass',   10**13.5),
        ('min_rotations',   3),
        ('max_rotations',   500),
        ('rot_template',    True),
        ('split_rot',       1),

        # MDPL
#         ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                      'h': 0.6777,
#                      'Omega_m': 0.307115,
#                      'Omega_l': 0.692885,
#                      'c': 299792.458  # [km/s]
#                      }),
        
#         # Uchuu
#         ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                      'h': 0.6774, # not used
#                      'Omega_m': 0.3089,
#                      'Omega_l': 0.6911,
#                      'c': 299792.458  # [km/s]
#                      }),
        # Magneticum
        ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
                     'h': 0.704, # not used
                     'Omega_m': 0.272,
                     'Omega_l': 0.728,
                     'c': 299792.458  # [km/s]
                     }),

        ('samp_hosts',      1),
        ('samp_gals',       1)
    ])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETER HANDLING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@func_stats
def handle_parameters():
    global debug, config, mock_tools
    
    if debug:
        print('>'*10, 'DEBUGGING', '<'*10)
        config['samp_hosts'] = 100
        config['samp_gals'] = 0.01
        config['dn_dlogm'] = 0
    
    print('\n~~~~~ GENERATING ' + config['catalog_name'] + ' ~~~~~~')

    for key in config.keys():
        print(key + ' : ' + str(config[key]))

    if config['cut_size'] == 'small':
        config['aperture'] = 1.1  # the radial aperture in comoving Mpc/h
        config['vcut'] = 1570.  # maximum velocity is vmean +/- vcut in km/s
    elif config['cut_size'] == 'medium':
        config['aperture'] = 1.6  # Mpc/h
        config['vcut'] = 2200.  # km/s
    elif config['cut_size'] == 'large':
        config['aperture'] = 2.3  # Mpc/h
        config['vcut'] = 3785.  # km/s
    else:
        raise Exception('Invalid cut_size')


    """
    Since d(z) has no simple, analytic form, but is smooth and continuous,
    we will sample it at many z and use scipy's interpolation function to
    reduce calculation time. Note, scipy interp is faster than numpy's
    """
    z_samp = np.arange(0, 5, 0.0001)
    d_samp = [d_from_z(i, config['cosmo']) for i in z_samp]

    # d(z) interpolation: Comoving distance from redshift
    mock_tools.d_from_z_sci = interp1d(z_samp, d_samp)

    # z(d) interpolation: Redshift from comoving distance
    mock_tools.z_from_d_sci = interp1d(d_samp, z_samp)

    """
    We'll assume that each host (cluster) is centered at z=config['z'].
    Therefore, comoving distances and Hubble recession velocities of the
    hosts are constant.
    """
    mock_tools.d_clu = d_from_z(config['z'], config['cosmo'])
    mock_tools.vH_clu = vH_from_z(config['z'], config['cosmo'])
    
    return config


# ~~~~~~~~~~~~~~~~~~~~~~~~ LOADING, PREPROCESSING DATA ~~~~~~~~~~~~~~~~~~~~~~~~

@func_stats
def load_and_preprocess():
    global config, host_data, gal_data
    
    print('\nLoading data...')
    host_data = load_raw(os.path.join(config['wdir'], config['in_folder'],
                                      config['host_file']))
    # only changes columns if they're misnamed
    host_data = host_data.rename(columns={'upid':'upId','id':'rockstarId'}, copy=False)

    if config['gal_file'] is None:
        gal_data = host_data
    else:
        gal_data = load_raw(os.path.join(config['wdir'], config['in_folder'],
                                         config['gal_file']))
        gal_data = gal_data.rename(columns={'upid': 'upId', 'mpeak': 'Macc',
                                            'mvir': 'Mvir'}, copy=False)

    # Cluster, galaxy constraints
    print('\nFinding hosts, galaxies...')
    host_data = host_data[(host_data['upId'] == -1) &
                          (host_data['M200c'] >= config['host_min_mass'])]

    gal_data = gal_data[(gal_data['upId'] != -1) &
                        (gal_data[config['gal_mass_def']] >= config['gal_min_mass'])]

    # Subsample
    if config['samp_hosts'] < 1:
        host_data = host_data.sample(int(config['samp_hosts'] * len(host_data)), replace=False)
    elif config['samp_hosts'] > 1:
        host_data = host_data.sample(int(config['samp_hosts']), replace=False)
        
    if config['samp_gals'] < 1:
        gal_data = gal_data.sample(int(config['samp_gals'] * len(gal_data)), replace=False)
    elif config['samp_gals'] > 1:
        gal_data = gal_data.sample(int(config['samp_gals']), replace=False)

    host_data = host_data.reset_index(drop=True)
    gal_data = gal_data.reset_index(drop=True)

    print('host_data length: ' + str(len(host_data)))
    print('gal_data length: ' + str(len(gal_data)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIND TRUE MEMBERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@func_stats
def find_true_members():
    global gal_data, host_members
    
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

def in_pad_region(key):
    global config, gal_data
    # Assign to padding regions. To be used in multiprocessing
    axes = ('x', 'y', 'z')
    out = np.ones(shape=(len(gal_data), len(axes)))

    for j in range(len(key)):
        if key[j] is True:
            out[:, j] = (gal_data[axes[j]] < config['pad_size'])
        elif key[j] is False:
            out[:, j] = (gal_data[axes[j]] > (config['box_length'] - config['pad_size']))

    return np.argwhere(np.sum(out, axis=1) == len(axes)).flatten()

def reassign_true(host_i):
    global host_data, gal_data, host_members, pad_gal_copies
    
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

@func_stats
def pad():
    global config, gal_data, host_members, pad_gal_copies
    
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
    config['pad_size'] = 1.0 * (config['vcut'] + max_gal_v + max_clu_v) / H_from_z(config['z'], config['cosmo'])

    config['box_length'] = config['volume']**(1./3)

    print('pad_size/box_length: ' + str(config['pad_size'] / config['box_length']))

    pad_regions = []
    pad_directions = (None, False, True)

    for i in pad_directions:
        for j in pad_directions:
            for k in pad_directions:
                pad_regions.append((i, j, k))

    _ = pad_regions.remove((None, None, None))

    axes = ('x', 'y', 'z')


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
                gal_data.loc[c: c_end - 1, axes[j]] += config['box_length']
            elif pad_regions[i][j] is False:
                gal_data.loc[c: c_end - 1, axes[j]] -= config['box_length']

        c = c_end

    print('Padded gal_data length: ' + str(len(gal_data)))

    
    """
    Host-galaxy relationships might take place over a periodic boundary.
    Therefore, we must check if any padded copies of galaxies are closer to
    their respective hosts than the original galaxies
    """

    print('\nReassigning true members to account for padding...')
#     with mp.Pool(processes=n_proc) as pool:
#         # each process checks a different padding region
#         reassignments = pool.map(reassign_true, range(len(host_data)))

#     for i in range(len(host_data)):
#         if i % (10**4) == 0:
#             print(str(int(i / (10**4))) + ' / ' +
#                   str(int(len(host_data) / (10**4))))
#         host_members[host_data.at[i, 'rockstarId']] = reassignments[i]
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~ REDUCING DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~
@func_stats
def reduce():
    global host_data, gal_data
    print('\nReducing dataset...')  # Dropping unnecessary data fields

    host_drop = ['row_id', 'upId', 'pId', 'descId', 'breadthFirstId', 'scale']
    gal_drop = []

    host_data = host_data[[i for i in host_data.columns.values if i not in
                           host_drop]]
    gal_data = gal_data[[i for i in gal_data.columns.values if i not in gal_drop]]

# ~~~~~~~~~~~~~~~~~~~~ CALCULATING BOOSTED ROTATION ANGLES ~~~~~~~~~~~~~~~~~~~~
@func_stats
def get_rot_assign():
    global config, host_data, rot_assign
    
    print('Calculating boosted rotation angles...')

    host_rots = pd.DataFrame(0, index=host_data.index,
                             columns=['logM', 'num_rot'])

    host_rots.loc[:, 'logM'] = np.log10(host_data['M200c'])
    host_rots.loc[:, 'num_rot'] = 0

    host_rots = host_rots.sort_values(by='logM')

    start = np.argwhere(host_rots['logM'].values > np.log10(config['boost_minmass'])).min()


    window = [0, 1]

    out = np.zeros(len(host_rots), dtype=int)
    for i in range(start, len(host_rots)):
        if i%1e4==0: print(i, '/', len(host_rots))

        while (host_rots['logM'].iloc[window[0]] < host_rots['logM'].iloc[i]
               - config['dlogm'] / 2.): 
            window[0] += 1
        while (window[1] < len(host_rots)):
            if(host_rots['logM'].iloc[window[1]] >= host_rots['logM'].iloc[i]
                    + config['dlogm'] / 2.):
                break
            window[1] += 1

        i_dn_dlogm = (window[1] - window[0]) / config['volume'] / config['dlogm']

        out[i] = int(config['dn_dlogm'] / i_dn_dlogm) + 1

    host_rots['num_rot'] = out

    host_rots.iloc[np.argwhere(host_rots['num_rot'].values < config['min_rotations']),
                   host_rots.columns.get_loc('num_rot')] = config['min_rotations']
    host_rots.iloc[np.argwhere(host_rots['num_rot'].values > config['max_rotations']),
                   host_rots.columns.get_loc('num_rot')] = config['max_rotations']

    rot_assign = pd.DataFrame(
        True, index=host_rots.index,
        columns=[(0, 0, 0), (np.pi / 2, 0, 0), (np.pi / 2, np.pi / 2, 0)])
    
    if config['rot_template']:
        ang_template = fibonacci_sphere(max(host_rots['num_rot']) - config['min_rotations'])

    for n_rot in host_rots['num_rot'].unique():
        ang_list = fibonacci_sphere(n_rot - config['min_rotations'])
        if (len(ang_list)>0) and config['rot_template']:
            ang_list = match_angles(ang_list, ang_template)

        for ang in ang_list:
            if (*ang, 0) in rot_assign.index:
                rot_assign[(*ang, 0)] = (rot_assign[(*ang, 0)]) | (host_rots['num_rot'] == n_rot)
            else:
                rot_assign[(*ang, 0)] = host_rots['num_rot'] == n_rot

    print('# Rotations:' + str(host_rots['num_rot'].unique()))


    # ~~~~~~~~~~~ DIVIDE ROTATION CALCULATIONS TO MULTIPLE PROCESSES ~~~~~~~~~~~~~~
    print('Dividing rotation angles over multiple processes...')

    # it takes about a 1000 (UM) or 7000 (Uchuu) kdtree queries to equal one kdtree generation

    if len(rot_assign.columns) < n_proc:
        critical_num = len(host_data)/(n_proc*config['split_rot'])
    else:
        critical_num = max(1000, len(host_data)/(n_proc*config['split_rot'])) 

    for i in range(len(rot_assign.columns)): # note range iterator is constant through loop
        ang = rot_assign.columns[i]
        num_in_rot = np.sum(rot_assign[ang])

        if num_in_rot > critical_num:
            tot_processes = int(num_in_rot/critical_num)+1

            ind_processes = np.argwhere(rot_assign[ang].values).flatten()

            proc_assign = np.random.choice(tot_processes, size=num_in_rot, replace=True)

            for j in range(tot_processes):
                rot_assign[(*(ang[:2]), j)] = False
                rot_assign[(*(ang[:2]), j)].iloc[ind_processes[proc_assign==j]] = True
        

    print(critical_num)
    print(rot_assign.sum(axis=0))
    


# ~~~~~~~~~~~~~~~~~~~ SAVE AND LOAD INTERMEDIATES ~~~~~~~~~~~~~~~~~~~
@func_stats
def save_intermediates():
    global config, host_data, gal_data, host_members, rot_assign
    
    cache_dir = os.path.join(config['wdir'],
                             config['out_folder'],
                             config['catalog_name'] + '.cache')
    
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    
    with open(os.path.join(cache_dir, 'cache.p'), 'wb') as f:
        pickle.dump(config, f)
    with open(os.path.join(cache_dir, 'host_members.p'), 'wb') as f:
        pickle.dump(dict(host_members), f)
    
    host_data.to_pickle(os.path.join(cache_dir, 'host_data.p'))
    gal_data.to_pickle(os.path.join(cache_dir, 'gal_data.p'))
    
    if config['split_rot']<=1:
        rot_assign.to_pickle(os.path.join(cache_dir, 'rot_assign.p'))
    else:
        print('\nDiving over multiple rot-assigns...')
        col_count = rot_assign.sum(axis=0)
        avg = np.sum(col_count)/config['split_rot']
        
        print('Average:', avg)
        
        j0=0
        ji=0
        j=0
        for i in range(config['split_rot']-1):
            ang_list = []
            while ji < (i+1)*avg:
                ang_list.append(j)
                ji += col_count.iloc[j]
                j+=1
                
            print('#Clusters:', (ji-j0), '; #Rotations:', len(ang_list))
            j0=ji
            
            x = rot_assign.iloc[:, ang_list]
            x.to_pickle(os.path.join(cache_dir, f'rot_assign_{i}.p'))
        
        x = rot_assign.iloc[:, j:]
        x.to_pickle(os.path.join(cache_dir, f'rot_assign_{i+1}.p'))
    
@func_stats    
def load_intermediates():
    global config, host_data, gal_data, host_members, rot_assign, load_ind
    
    cache_dir = os.path.join(config['wdir'],
                             config['out_folder'],
                             config['catalog_name'] + '.cache')
    
    with open(os.path.join(cache_dir, 'cache.p'), 'rb') as f:
        config = pickle.load(f)
    with open(os.path.join(cache_dir, 'host_members.p'), 'rb') as f:
        host_members = pickle.load(f)
    
    host_data = pd.read_pickle(os.path.join(cache_dir, 'host_data.p'))
    gal_data = pd.read_pickle(os.path.join(cache_dir, 'gal_data.p'))
    
    if load_ind==None:
        rot_assign = pd.read_pickle(os.path.join(cache_dir, 'rot_assign.p'))
    else:
        rot_assign = pd.read_pickle(os.path.join(cache_dir, f'rot_assign_{load_ind}.p'))

# ~~~~~~~~~~~~~~~~~~~~~~~~ GENERATE PROJECTED KDTREES ~~~~~~~~~~~~~~~~~~~~~~~~
def grow_tree(angle):
    R = rot_matrix_LOS(*angle)

    gal_pos_rot = np.matmul(gal_data[['x', 'y', 'z']].values, R.T)
    
    return KDTree(gal_pos_rot[:,:2], leafsize=int(10**3.5))

@func_stats
def generate_forest(ang_list):
    global rot_assign, trees, tree_pool
    
    print('Growing KDTrees...')
    
    print('Total rotations:', len(ang_list))
    if tree_pool:
        with mp.Pool(processes=n_proc) as pool:
            tree_list = pool.map(grow_tree, ang_list, chunksize=1)
    else:
        tree_list = list(map(grow_tree, ang_list))

    trees = dict(zip(ang_list, tree_list))
    
# ~~~~~~~~~~~~~~~~~~~ CALCULATING ROTATED MOCK OBSERVATIONS ~~~~~~~~~~~~~~~~~~~

def cut_mock(angle_id=None):
    global config, host_data, gal_data, host_members, rot_assign
    
    """
    Given a set of rotation angles, rotate catalog. Iterate through clusters.
    Cut a long cylinder through the entire sim around projected cluster center
    (called pillars). Calculate all vlos. Create pure, contaminated catalogs.

    This is wrapped in a function so it can be parallelized.
    
    I removed virial true members (true members are <Rproj) after commit
    4b1c55b0a65e794f2cbc295be4268edd7b987d1d.
    """

    angle = angle_id[:2]
    
    # ~~~~~ rotate galaxy-host positions+velocities ~~~~~
    R = rot_matrix_LOS(*angle)
    
    cl_list = rot_assign.index.values[rot_assign[angle_id]] # cluster indices to be cut
#     print(angle_id, len(cl_list))
    

    gal_rot = pd.DataFrame(
        np.concatenate((np.matmul(gal_data[['x', 'y', 'z']].values, R.T),
                        np.matmul(gal_data[['vx', 'vy', 'vz']].values, R.T)),
                       axis=1), columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    host_rot = pd.DataFrame(
        np.concatenate((np.matmul(host_data[['x', 'y', 'z']].values, R.T),
                        np.matmul(host_data[['vx', 'vy', 'vz']].values, R.T)),
                       axis=1), columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
#     print('calc', angle_id)

    # ~~~~~ initialize projected KDTree ~~~~~
    # 10**3.5 found to be optimal empirically
    gal_tree_proj = trees[angle] # KDTree(gal_rot[['x', 'y']], leafsize=int(10**3.5))

    # ~~~~~ initialize catalogs ~~~~~
    # empty catalog files
    pure_catalog = Catalog(prop=host_data, gal=[])
    contam_catalog = Catalog(prop=host_data, gal=[])

    # record which clusters were successfully measured
    pure_ind = []
    contam_ind = []

    # output member galaxy information
    gal_prop = [(config['gal_mass_def'], '<f8'), ('Mvir', '<f8'), ('Macc', '<f8')] # 
    
    if config['gal_file'] is not None:
        gal_prop = [('obs_sm', '<f8'), ('obs_sfr', '<f8'), ('id', '<i8')]
    else:
        gal_prop += [('rockstarId', '<i8')]# , ('Vmax', '<f8'), ('Vpeak', '<f8')]
        
    gal_prop = [('m', '<f8'), ('id', '<i8')]

    dtype_gal = [('xproj', '<f8'), ('yproj', '<f8'), ('zproj', '<f8'),
                 ('Rproj', '<f8'), ('vlos', '<f8'),
                 ('memb_halo', '?'),
                 *gal_prop
                 ]
    
    # ~~~ iterate through hosts, assign member galaxies, calculate observables
    
    mem_proj_list = gal_tree_proj.query_ball_point(
        host_rot.loc[cl_list, ['x', 'y']], config['aperture'])

    for i in range(len(cl_list)):  # for each cluster
        ind = cl_list[i]
        
        if i%1000==0:
            print(angle_id, 'ind', ind)

        # Grab galaxy members as assigned by halo algo
        try:
            mem_true_ind = host_members[host_data.loc[ind, 'rockstarId']]
        except KeyError:
            print('error:',ind)
            continue

        # Sample KDTree to find potential projected members
        mem_proj_ind = mem_proj_list[i]

        # calculate relative velocities of all potential projected members
        v_rel_proj = cal_v_rel(host_rot.loc[ind, 'z'],
                               host_rot.loc[ind, 'vz'],
                               gal_rot.loc[mem_proj_ind, 'z'],
                               gal_rot.loc[mem_proj_ind, 'vz'],
                               config['cosmo'])

        # restrict projected member list to those with |v_los|<v_cut
        v_rel_proj = v_rel_proj[np.abs(v_rel_proj) < config['vcut']]
        mem_proj_ind = v_rel_proj.index.to_series()

        # contaminated catalog
        if len(mem_proj_ind) > config['min_richness']:

            # initialize array of output member galaxy information
            clu_gals = np.zeros(shape=(len(mem_proj_ind)),
                                dtype=dtype_gal)

            # assign relevant information
            clu_gals[['xproj', 'yproj', 'zproj']] = (
                gal_rot.loc[mem_proj_ind, ['x', 'y', 'z']]
                - host_rot.loc[ind, ['x', 'y', 'z']]).to_records(index=False)
            clu_gals['Rproj'] = np.sqrt(
                clu_gals['xproj']**2 + clu_gals['yproj']**2)

            clu_gals['vlos'] = v_rel_proj

            clu_gals['memb_halo'] = [x in mem_true_ind for x in mem_proj_ind]
            
            clu_gals[[p[0] for p in gal_prop]] = gal_data.loc[
                mem_proj_ind, [p[0] for p in gal_prop]].to_records(index=False)

            # append output galaxy information to catalog
            contam_catalog.gal.append(clu_gals)
            contam_ind.append(ind)

        # pure catalog
        if len(mem_true_ind) > config['min_richness']:

            # calculate v_los for true members which may not fall in projected
            # cylinder
            mem_truenotproj_ind = [
                j for j in mem_true_ind if j not in mem_proj_ind]

            v_rel_truenotproj = cal_v_rel(host_rot.loc[ind, 'z'],
                                          host_rot.loc[ind, 'vz'],
                                          gal_rot.loc[
                mem_truenotproj_ind, 'z'],
                gal_rot.loc[
                mem_truenotproj_ind, 'vz'],
                config['cosmo'])

            v_rel_true = v_rel_proj.append(v_rel_truenotproj)[mem_true_ind]

            # initialize array of output member galaxy information
            clu_gals = np.zeros(shape=(len(mem_true_ind)),
                                dtype=dtype_gal)

            # assign relevant information
            clu_gals[['xproj', 'yproj', 'zproj']] = (
                gal_rot.loc[mem_true_ind, ['x', 'y', 'z']]
                - host_rot.loc[ind, ['x', 'y', 'z']]).to_records(index=False)
            clu_gals['Rproj'] = np.sqrt(
                clu_gals['xproj']**2 + clu_gals['yproj']**2)

            clu_gals['vlos'] = v_rel_true

            clu_gals[[p[0] for p in gal_prop]] = gal_data.loc[
                mem_true_ind, [p[0] for p in gal_prop]].to_records(index=False)

            clu_gals['memb_halo'] = True

            # append output galaxy information to catalog
            pure_catalog.gal.append(clu_gals)
            pure_ind.append(ind)

    # restrict output catalogs to only correctly calulated clusters (e.g.
    # above min_richness)
    pure_catalog.prop = pure_catalog.prop.iloc[pure_ind]
    contam_catalog.prop = contam_catalog.prop.iloc[contam_ind]

    # delete unnecessary data
    del gal_rot
    del host_rot

#     print('end', angle_id)
    return (pure_catalog, contam_catalog)

@func_stats
def generate_catalogs(ang_ids):
    
    print(f'Cutting mock clusters with {n_proc} processes...')
    with mp.Pool(processes=n_proc) as pool:
        catalogs = list(pool.map(cut_mock, ang_ids, chunksize=1))
        
    return catalogs

@func_stats
def consolidate_catalogs(catalogs, ang_seq):
    
    pure_len = 0
    contam_len = 0

    for i in range(len(catalogs)):
        pure_len += len(catalogs[i][0].prop)
        contam_len += len(catalogs[i][1].prop)

    print('\nCombining rotation catalogs...')

    # Initialize catalogs

    cat_par = OrderedDict([
        ('catalog_name',   config['catalog_name']),

        ('z',   config['z']),

        ('min_mass',   config['host_min_mass']),
        ('min_richness',   config['min_richness']),

        ('aperture',   config['aperture']),
        ('vcut',   config['vcut']),

        ('cosmo',   config['cosmo']),
        ('config',  config)
    ])


    contam_catalog = Catalog(
        par=cat_par.copy(),
        prop=pd.DataFrame(np.zeros(
            shape=(contam_len, catalogs[0][1].prop.shape[1] + 2)),
            columns=np.append(catalogs[0][1].prop.columns.values, ['rot_theta', 'rot_phi'])),
        gal=[None] * contam_len)

    cat_par['aperture'] = None
    cat_par['vcut'] = None

    pure_catalog = Catalog(
        par=cat_par.copy(),
        prop=pd.DataFrame(np.zeros(
            shape=(pure_len, catalogs[0][0].prop.shape[1] + 2)),
            columns=np.append(host_data.columns.values, ['rot_theta', 'rot_phi'])),
        gal=[None] * pure_len)


    pure_c = 0
    contam_c = 0

#     rot_id_dict = {}

    for i in range(len(catalogs)):
        ang = ang_seq[i]
#         if ang[-1]==0:
#             rot_num = len(rot_id_dict)
#             rot_id_dict[ang[:2]] = rot_num
#         else:
#             rot_num = rot_id_dict[ang[:2]]

        pure, contam = catalogs[i]

        pure_c_end = pure_c + len(pure.prop)
        contam_c_end = contam_c + len(contam.prop)

        pure_catalog.prop.iloc[pure_c:pure_c_end, :-2] = pure.prop.values
        pure_catalog.prop.loc[contam_c:contam_c_end, 
                              ['rot_theta','rot_phi']] = ang[:2]
        pure_catalog.gal[pure_c:pure_c_end] = pure.gal

        contam_catalog.prop.iloc[contam_c:contam_c_end, :-2] = contam.prop.values
        contam_catalog.prop.loc[contam_c:contam_c_end, 
                                ['rot_theta','rot_phi']] = ang[:2]
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
    
    return pure_catalog, contam_catalog


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAVE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@func_stats
def save_data(pure_catalog, contam_catalog):
    global config, debug, load_ind
    
    if not debug:
        print('Saving...')
        
        if load_ind is None:
            pure_catalog.save(os.path.join(config['wdir'],
                                           config['out_folder'],
                                           config['catalog_name'] + '_pure.p'))
            contam_catalog.save(os.path.join(config['wdir'],
                                             config['out_folder'],
                                             config['catalog_name'] + '_contam.p'))
        else:
            pure_catalog.save(os.path.join(config['wdir'],
                                           config['out_folder'],
                                           config['catalog_name'] + '.cache',
                                           config['catalog_name'] + f'_pure_{load_ind}.p'))
            contam_catalog.save(os.path.join(config['wdir'],
                                             config['out_folder'],
                                             config['catalog_name'] + '.cache',
                                             config['catalog_name'] + f'_contam_{load_ind}.p'))



config, host_data, gal_data, host_members, rot_assign, trees, pad_gal_copies = [None]*7
if __name__=='__main__':
    
    t0 = time.time()
    
    build_config()
    
    if (not load_cache) or (save_cache and load_cache):
        handle_parameters()

        load_and_preprocess()

        find_true_members()

        pad()
    
        reduce()

        get_rot_assign()
        
        if save_cache and not debug:
            save_intermediates()
        
    if load_cache:
        load_intermediates()
        
        if not save_cache:
            handle_parameters()
    
    ang_list = np.unique([x[:2] for x in rot_assign.columns.values], axis=0)
    ang_list = [tuple(x) for x in ang_list]
    ang_seq = []
    
    if (len(ang_list) > 5) & (load_ind != None) :
        print('Segmenting ang_list...')
        catalogs = []
        for i in range(0,len(ang_list),3):
            print(i, '/', len(ang_list))
            generate_forest(ang_list[i:i+3])
            
            ang_ids = [x for x in rot_assign.columns.values if tuple(x[:2]) in trees]
            catalogs += generate_catalogs(ang_ids)
            ang_seq += ang_ids
    else:
        generate_forest(ang_list)
        catalogs = generate_catalogs(rot_assign.columns.values)
        ang_seq = rot_assign.columns.values
        
    pure_catalog, contam_catalog = consolidate_catalogs(catalogs, ang_seq)
    
    save_data(pure_catalog, contam_catalog)
    
    print('\nAll done!')
    print('Total Runtime: ' + str((time.time() - t0) / 60.) + ' minutes')

"""
    # ~~~~ Previously run mocks ~~~~

    # # UM with z=0.117
    # ('catalog_name',    'Rockstar_UM_z=0.117'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:120_v3.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.895100.npy'),
    # ('z',               0.117),

    # # UM with z=0.000
    # ('catalog_name',    'Rockstar_UM_z=0.000'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:125.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_1.000000.npy'),
    # ('z',               0.),

    # # UM with z=0.022->0.0231
#     ('catalog_name',    'Rockstar_UM_z=0.0.022-0231_SMCut=9.5'),
#     ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#     ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#     ('z',               0.0231), # coma

    # # UM with z=0.394
    # ('catalog_name',    'Rockstar_UM_z=0.394'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:110.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.717300.npy'),
    # ('z',               0.394),

    # # UM with z=0.045
    # ('catalog_name',    'Rockstar_UM_z=0.045'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:123.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.956700.npy'),
    # ('z',               0.045),

    # # UM with z=0.194
    # ('catalog_name',    'Rockstar_UM_z=0.194'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:117.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.837600.npy'),
    # ('z',               0.194),

    # # UM with z=0.248
    # ('catalog_name',    'Rockstar_UM_z=0.248'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:115.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.801300.npy'),
    # ('z',               0.248),

    # # UM with z=0.304
    # ('catalog_name',    'Rockstar_UM_z=0.304'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:113.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.766600.npy'),
    # ('z',               0.304),

    # UM with z=0.364
#     ('catalog_name',    'Rockstar_UM_z=0.364'),
#     ('host_file',       'cosmosim/MDPL2_Rockstar_snap:111.csv'),
#     ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.733300.npy'),
#     ('z',               0.364),

    # # UM with z=0.425
    # ('catalog_name',    'Rockstar_UM_z=0.425'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:109.csv'),
    # ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.701600.npy'),
    # ('z',               0.425),

    # MD with z=0.117
    # ('catalog_name',    'Rockstar_MD_z=0.117'),
    # ('host_file',       'cosmosim/MDPL2_Rockstar_snap:120_v3.csv'),
    # ('gal_file',        None),
    # ('z',               0.117),

    # BigMDPL with z=0.1131
    # ('catalog_name',    'Rockstar_BigMDPL_z=0. 1131'),
    # ('host_file',       'cosmosim/BigMDPL_Rockstar_snap:73.csv'),
    # ('gal_file',        None),
    # ('z',               0.1131),
    
    # Uchuu with z=0.000
    # ('catalog_name',    'Uchuu_z=0.000_bigvol'),
    # ('host_file',       'uchuu/Uchuu_snap:050_m200c1e11'),
    # ('gal_file',        None),
    # ('z',               0.),

    # Uchuu with z=0.0231
    # ('catalog_name',    'Uchuu_z=0.0231_bigvol'),
    # ('host_file',       'uchuu/Uchuu_snap:050_m200c1e11'),
    # ('gal_file',        None),
    # ('z',               0.0231), # coma
    
    # Uchuu with z=0.000->0.0231 and Macc>=1e11
#     ('catalog_name',    'Uchuu_z=0.0-0.231_Macc1e11'),
#     ('host_file',       'uchuu/Uchuu_z=0.0_Maccg=11.csv'),
#     ('gal_file',        None),
#     ('z',               0.0231), # coma

"""

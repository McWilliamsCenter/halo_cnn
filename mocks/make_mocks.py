"""
interact --ntasks-per-node=28 -t 08:00:00

module load anaconda3/5.1.0
source activate jupy
cd halo_cnn/
python ./catalog/make_catalog.py
"""



## ~~~~~ IMPORTS ~~~~~~
import sys
import os
import numpy as np
import pandas as pd
import time
import pickle

import scipy.integrate as integrate
import multiprocessing as mp
from scipy.spatial import KDTree
from collections import OrderedDict, defaultdict

from tools.catalog import Cluster, Catalog


## ~~~~~~ PARAMETERS ~~~~~~
par = OrderedDict([ 
    ('catalog_name' ,   'Rockstar_UM_z=0.117'),
    
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn'),
    ('in_folder'    ,   'data_raw'),
    ('out_folder'   ,   'data_mocks'),
    
    ('host_file'    ,   'MDPL2_Rockstar_snap:120.csv'),
    ('gal_file'     ,   'sfr_catalog_0.895100.npy'),
    
    ('z'            ,   0.117),
    
    ('min_mass'     ,   10**(13.5)),
    ('min_richness' ,   10),
    
    ('cut_size'     ,   'large'),
    
    ('cosmo' ,   {'H_0': 100, # [km/s/(Mpc/h)]
                  'Omega_m': 0.307115,
                  'Omega_l': 0.692885,
                  'c': 299792.458 # [km/s]
                 })
    ])

debug = False
## ~~~~~ PARAMETER HANDLING ~~~~~

print('\n~~~~~ GENERATING ' + par['catalog_name'] + ' ~~~~~~')

for key in par.keys():
    print(key + ' : ' + str(par[key]))

cosmo = par['cosmo']

if par['cut_size'] == 'small':
    aperture = 1.1     #the radial aperture in comoving Mpc/h
    vcut = 1570.       #maximum velocity is vmean +/- vcut in km/s
    
elif par['cut_size'] == 'medium':
    aperture = 1.6 #Mpc/h
    vcut = 2500. #km/s
    
elif par['cut_size'] == 'large':
    aperture = 2.3 #Mpc/h
    vcut = 3785. #km/s
else:
    raise Exception('Invalid cut_size')


## ~~~~~~ UTILITY ~~~~~~
def load_raw(filename):
    print('Loading data from: ' + filename)
    if filename[-4:] == '.csv':
        return pd.read_csv(filename, index_col=0)
    elif filename[-4:] == '.npy':
        return pd.DataFrame(np.load(filename))
    else:
        print('Invalid file type: ' + filename[-4:])
        
        
## ~~~~~ COSMOLOGY FUNCTIONS ~~~~~~
def H_from_z(z, cosmo):
    H = cosmo['H_0'] * (cosmo['Omega_m']*(1 + z)**3 + cosmo['Omega_l'])**0.5
    return H

def d_from_z(z, cosmo):
    d = integrate.quad(lambda x: cosmo['c']/H_from_z(x, cosmo), 0, z)
    return d[0]

z_samp = np.arange(0,5,0.0001)
d_samp = [d_from_z(i, cosmo) for i in z_samp]

def d_from_z_nump(z):
    d_nump = np.interp(z, z_samp, d_samp)
    return d_nump
def z_from_d_nump(d):
    z_nump = np.interp(d, d_samp, z_samp)
    return z_nump

def vH_from_z(z, cosmo):
    vH = cosmo['c']*((1+z)**2 - 1)/((1+z)**2 + 1)
    return vH

def add_v(v1, v2, cosmo):
    v_sum = (v1 + v2)/(1 + v1*v2/cosmo['c']**2)
    return v_sum
def sub_v(v1, v2, cosmo):
    v_diff = (v1 - v2)/(1 - v1*v2/cosmo['c']**2)
    return v_diff

d_clu = d_from_z(par['z'], cosmo)
vH_clu = vH_from_z(par['z'], cosmo)

def cal_v_rel(host_pos, host_v, gal_pos, gal_v, cosmo):
    dist_gal_clu = gal_pos-host_pos

    d_gal = d_clu + dist_gal_clu
    
    z_gal = z_from_d_nump(d_gal)
    
    vH_gal = vH_from_z(z_gal, cosmo)
    
    v_gal = add_v(vH_gal, gal_v, cosmo)
    
    v_clu = add_v(vH_clu, host_v, cosmo)
    
    v_rel = sub_v(v_gal, v_clu, cosmo)
    
    return v_rel


t0 = time.time()

## ~~~~~ LOADING, PREPROCESSING DATA ~~~~~
print('\nLoading data...')
host_data = load_raw(os.path.join(par['wdir'], par['in_folder'], par['host_file']))

if par['gal_file'] is None:
    gal_data = host_data
else:
    gal_data = load_raw(os.path.join(par['wdir'], par['in_folder'], par['gal_file']))
    

# Cluster, galaxy constraints
print('\nFinding hosts, galaxies...')
host_data = host_data[  (host_data['upId']==-1) & 
                        (host_data['M200c'] > par['min_mass'])]
                        
gal_data = gal_data[gal_data['upid']!=-1]

if debug:
    host_data = host_data.sample(10**4)
    gal_data = gal_data.sample(10**5)

host_data = host_data.reset_index(drop=True)
gal_data = gal_data.reset_index(drop=True)

print('host_data length: ' + str(len(host_data)))
print('gal_data length: ' + str(len(gal_data)))

## Iterate through galaxies, find true_members and pad gal_data

host_members= defaultdict(lambda:[]) # Relates host rockstarid to indices of member galaxies

print('\nAssigning true members...')

for i in range(len(gal_data)):
    if i%(10**5)==0 :print(str(int(i/(10**5))) + ' / ' + str(int(len(gal_data)/(10**5))))
    # Assign to host

    host_members[gal_data.loc[i,'upid']].append(i)

print('\nReducing dataset...')

host_drop = ['row_id','upId','pId','descId','breadthFirstId','scale']
gal_drop = []

host_data = host_data[[i for i in host_data.columns.values if i not in host_drop]]
gal_data = gal_data[[i for i in gal_data.columns.values if i not in gal_drop]]



## ~~~~~ PADDING DATA ~~~~~~
"""
to_pad is a dictionary which relates tuples to an array of galaxy indices. The tuples have length 3 and correspond to the x,y,z axes, respectively. 

For a given axis n, if the tuple value is 'True', the galaxies within the related array will have their 'n'-axis added one box_length when padding. If the tuple value is 'False', the galaxies within the related array will have their 'n'-axis subtracted one box_length when padding. If 'None' appears, the 'n'-axis is left alone while padding.

I did this to automate the padding process and make it O(n)
"""

print('\nDetermining padding regions...')
pad_size = 1.05*(vcut + np.sqrt(gal_data['vx']**2 + gal_data['vy']**2 + gal_data['vz']**2).max())/100.# CHECK THIS
box_length = 1000

pad_regions = []
pad_directions = (None, False, True)

for i in pad_directions:
    for j in pad_directions:
        for k in pad_directions:
            pad_regions.append((i,j,k))
            
_ = pad_regions.remove((None,None,None))

axes = ('x','y','z')

def in_pad_region(key):
    # Assign to padding regions
    p_region = []
    
    for i in range(len(gal_data)):
        in_region = True
            
        for j in range(len(key)):
            if key[j]==True:
                if gal_data.iloc[i][axes[j]] > pad_size:
                    in_region=False
                    break
            elif key[j]==False:
                if gal_data.iloc[i][axes[j]] < (box_length - pad_size):
                    in_region=False
                    break
        
        if in_region:
            p_region.append(i)
            
    # print(key, len(p_region))
            
    return p_region

with mp.Pool(processes=13) as pool:
    to_pad_ind = pool.map(in_pad_region, pad_regions)

print('Padding regions: ')
for i in range(len(pad_regions)):
    s = ''
    for j in range(len(pad_regions[i])):
        if pad_regions[i][j] == True:
            s += axes[j] + '+ '
        elif pad_regions[i][j] == False:
            s += axes[j] + '- '
    
    print(s + ' : ' + str(len(to_pad_ind[i])))
            
# pad_regions now relates padding directions to which galaxies are to be padded


num_padded = np.sum([len(to_pad_ind[i]) for i in range(len(to_pad_ind))])

pad_gal_data = pd.DataFrame(np.zeros(shape=(num_padded, gal_data.shape[1])), columns = gal_data.columns)

print('\nPadding...')
c = 0
for i in range(len(pad_regions)):
    print(str(i + 1) + ' / ' + str(len(pad_regions)))
    
    c_end = c + len(to_pad_ind[i])
    pad_gal_data.values[c : c_end,:] = gal_data.iloc[to_pad_ind[i]].values
        
    for j in range(len(pad_regions[i])):
        if pad_regions[i][j]==True:
             pad_gal_data.loc[c : c_end, axes[j]] += box_length
        elif pad_regions[i][j]==False:
             pad_gal_data.loc[c : c_end, axes[j]] -= box_length
        
    c = c_end

gal_data = gal_data.append(pad_gal_data, ignore_index=True)

print('Padded gal_data length: ' + str(len(gal_data)))

# gal_data is now padded.


## ~~~~~ CALCULATING ROTATED MOCK OBSERVATIONS ~~~~~~
def rot_matrix_rand(seed=124987):
    np.random.seed(seed)
    
    th_x, th_y, th_z = 2*np.pi*np.random.random(3)
    
    print('\nAngles: ' + str((th_x,th_y,th_z)))

    R_x = np.array([[1,0,0],
                    [0, np.cos(th_x), -np.sin(th_x)], 
                    [0, np.sin(th_x), np.cos(th_x)]]
                  )

    R_y = np.array([[np.cos(th_y), 0, np.sin(th_y)],
                    [0, 1, 0],
                    [-np.sin(th_y), 0, np.cos(th_y)]]
                  )

    R_z = np.array([[np.cos(th_z), -np.sin(th_z), 0],
                    [np.sin(th_z), np.cos(th_z), 0],
                    [0,0,1]]
                  )
    R = np.matmul(R_z, R_y, R_x)
    
    return R

def cut_mock(rot_seed=89847):
    # print('\nRotating Box')

    R = rot_matrix_rand(seed=rot_seed)

    gal_pos = pd.DataFrame(np.matmul(gal_data[['x','y','z']].values, R.T),
                           columns=['x','y','z'])
    gal_v = pd.DataFrame(np.matmul(gal_data[['vx','vy','vz']].values, R.T),
                           columns=['vx','vy','vz'])
    host_pos = pd.DataFrame(np.matmul(host_data[['x','y','z']].values, R.T),
                           columns=['x','y','z'])
    host_v = pd.DataFrame(np.matmul(host_data[['vx','vy','vz']].values, R.T),
                           columns=['vx','vy','vz'])

    # Create KDTree of x,y positions
    # print('\nInitializing KDTree...')
    gal_tree = KDTree(gal_pos[['x','y']], leafsize=50)

    """
    Iterate through clusters. Cut cylinder through whole sim. Calculate all vlos. Create pure, contaminated catalogs. [TO-DO]
    cut a long cylinder through the entire sim around projected cluster center. I'll call these pillars.
    """
    # print('Sampling KDTree...')
    pillar_ind = gal_tree.query_ball_point(host_pos[['x','y']], aperture)

    pure_catalog = Catalog(prop = host_data,
                           gal = []
                          )
    contam_catalog = Catalog(prop=host_data,
                             gal=[]
                            )
    pure_ind = []
    contam_ind = []

    gal_dtype = [ ('x_proj','<f4'),
                  ('y_proj','<f4'),
                  ('v_los', '<f4'),
                  ('true_memb','<i4'),
                  ('mvir','<f4')
                ]
    gal_index = gal_data.index.to_series()


    for i in range(len(host_data)):
        if i%(10**4)==0 :print(str(int(i/(10**4))) + ' / ' + str(int(len(host_data)/(10**4) )))
        
        # calculate relative velocities
        v_rel_pillar = cal_v_rel( host_pos.iloc[i]['z'],
                                  host_v.iloc[i]['vz'],
                                  gal_pos['z'].iloc[pillar_ind[i]],
                                  gal_v['vz'].iloc[pillar_ind[i]],
                                  cosmo)
                                  

        obs_members_v_rel = v_rel_pillar[np.abs(v_rel_pillar) < vcut]
        
        true_members = host_members[host_data.iloc[i]['rockstarId']]
        
        # contaminated catalog   
        if len(obs_members_v_rel) > par['min_richness']:
            clu_gals = np.zeros(shape=(len(obs_members_v_rel)),
                                dtype = gal_dtype)
                                
            obs_members_index = obs_members_v_rel.index.to_series()
            
            clu_gals['x_proj'] = gal_pos['x'].loc[obs_members_index] - host_pos.iloc[i]['x']
            clu_gals['y_proj'] = gal_pos['y'].loc[obs_members_index] - host_pos.iloc[i]['y']
            clu_gals['v_los'] = obs_members_v_rel
            clu_gals['true_memb'] = [x in true_members
                                     for x in obs_members_index
                                    ]
            clu_gals['mvir'] = gal_data['mvir'].loc[obs_members_index]
            
            contam_catalog.gal.append(clu_gals)
            contam_ind.append(i)                          
        
        
        
        # pure catalog
        if len(true_members) > par['min_richness']:
            true_members_out_of_pillar = [j for j in true_members if j not in pillar_ind[i]]
            
            v_rel_out_of_pillar = cal_v_rel( host_pos.iloc[i]['z'],
                                             host_v.iloc[i]['vz'],
                                             gal_pos['z'].iloc[true_members_out_of_pillar],
                                             gal_v['vz'].iloc[true_members_out_of_pillar],
                                             cosmo)
                                             
            v_rel = v_rel_pillar.append(v_rel_out_of_pillar)

            clu_gals = np.zeros(shape=(len(true_members)),
                                dtype = gal_dtype)
            
            clu_gals['x_proj'] = gal_pos['x'].iloc[true_members] - host_pos.iloc[i]['x']
            clu_gals['y_proj'] = gal_pos['y'].iloc[true_members] - host_pos.iloc[i]['y']
            clu_gals['v_los'] = v_rel.loc[gal_index.iloc[true_members]]
            clu_gals['true_memb'] = True
            clu_gals['mvir'] = gal_data['mvir'].loc[true_members]
            
            pure_catalog.gal.append(clu_gals)
            pure_ind.append(i)


    # print('# pure catalog: ' + str(len(pure_ind)))
    # print('# contaminated catalog: ' + str(len(contam_ind)))

    pure_catalog.prop = pure_catalog.prop.iloc[pure_ind]
    contam_catalog.prop = contam_catalog.prop.iloc[contam_ind]
    
    return (pure_catalog, contam_catalog)


print('\nCutting cylinders...')

with mp.Pool(processes=13) as pool:
    catalogs = pool.map(cut_mock, 
                        np.random.randint(0,10**5,10)
                       )
print('Done.')

pure_len = 0
contam_len = 0

for i in range(len(catalogs)):
    print('\nRotation #' + str(i))
    print('pure len: ' + str(len(catalogs[i][0].prop)))
    print('contam len: ' + str(len(catalogs[i][1].prop)))
    
    pure_len += len(catalogs[i][0].prop)
    contam_len += len(catalogs[i][1].prop)

print('\nCombining rotation catalogs...')
pure_catalog = Catalog(prop = pd.DataFrame( np.zeros(shape=(pure_len, host_data.shape[1]+1)),
                                            columns = np.append(host_data.columns.values,'rotation')
                                          ),
                       gal = [None]*pure_len
                      )
contam_catalog = Catalog(prop = pd.DataFrame( np.zeros(shape=(contam_len, host_data.shape[1]+1)),
                                            columns = np.append(host_data.columns.values,'rotation')
                                          ),
                       gal = [None]*contam_len
                      )
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
    

## ~~~~~ SAVE DATA ~~~~~~

print('Saving...')
pure_catalog.save(os.path.join(par['wdir'], 
                               par['out_folder'], 
                               par['catalog_name'] + '_pure.p'))
contam_catalog.save(os.path.join(par['wdir'], 
                                 par['out_folder'], 
                                 par['catalog_name'] + '_contam.p'))


print('\nAll done!')
print('Runtime: ' + str((time.time()-t0)/60.) + ' minutes')


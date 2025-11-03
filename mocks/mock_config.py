from collections import OrderedDict

# new_config = OrderedDict([
#     # UM Macc cut with z=0.022->0.0231
#     ('catalog_name',    'Rockstar_UM_z=0.0.022-0231_Macc1e11-large-fix'),
#     ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#     ('gal_file',        None), # 'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#     ('z',               0.0231), # coma

#     ('volume', 1000.**3),

#     # MDPL
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.6777,
#                  'Omega_m': 0.307115,
#                  'Omega_l': 0.692885,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

# new_config = OrderedDict([
# #         # UM SM cut with z=0.022->0.0231
#     ('catalog_name',    'MDPL2_UM_z=0.0.022-0231_SM9.5-Xlarge-101321'),
#     ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#     ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#     ('z',               0.0231), # coma
    
#     ('dn_dlogm',        10.**-5.2), # -5.2
#     ('dlogm',           0.005),
#     ('boost_minmass',   10**13.),
#     ('min_rotations',   3),
#     ('max_rotations',   500),
    
#     ('volume', 1000.**3),

#     ('cut_size',        'extralarge'),
#     ('split_rot',       10),

#     # MDPL
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.6777,
#                  'Omega_m': 0.307115,
#                  'Omega_l': 0.692885,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

# new_config = OrderedDict([
#     # Uchuu with z=0.022->0.0231 and Macc>=1e11
#     ('catalog_name',    'Uchuu_z=0.022-0.231_VmaxP2.0-large-oldcorrect'),
#     ('host_file',       'uchuu/Uchuu_snap:049_m200c1e11.csv'),
#     ('gal_file',        None),
#     ('z',               0.0231), # coma

#     ('volume', 2000.**3),

#     # Uchuu
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.6774, # not used
#                  'Omega_m': 0.3089,
#                  'Omega_l': 0.6911,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

# new_config = OrderedDict([
#     # Magneticum with z=0.066340191 and m>=1e10.5
#     ('catalog_name',    'Magneticum_z=0.066_m1e11-large'),
#     ('host_file',       'magneticum/magn_snap136_cluster_fix+Dhayaa.csv'),        
#     ('gal_file',        'magneticum/magn_snap136_galaxies_fix.csv'),
#     ('z',               0.066340191),

#     ('volume', 352.**3),
#     ('min_rotations',   1),
#     ('dn_dlogm', 0),

#     # Magneticum
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.704, # not used
#                  'Omega_m': 0.272,
#                  'Omega_l': 0.728,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

# new_config = OrderedDict([    
#     # IllustriusTNG with z=0. and M*>=9.5
#     ('catalog_name',    'TNG300-1-z0p00-large'),
#     ('host_file',       'illustrius/TNG300-1_VelDispML_z0p00_Halos.csv'),
#     ('gal_file',        'illustrius/TNG300-1_VelDispML_z0p00_Subhalos.csv'),
#     ('z',               0.0),

#     ('volume', 302.6**3),

#     # Illustrius
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.704, # not used
#                  'Omega_m': 0.272,
#                  'Omega_l': 0.728,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

# new_config = OrderedDict([
#     # Magneticum with variable z and m_star>=9. Only z-direction
#     ('catalog_name',    'Magneticum_sn116_z=0.25_LOS-z'),
#     ('host_file',       'magneticum/raw_catalogs_fixed/magn_clu_sn116_z0.25208907.csv'),        
#     ('gal_file',        'magneticum/raw_catalogs_fixed/magn_gal_sn116_z0.25208907.csv'),
#     ('z',               0.25208907),

#     ('volume', 352.**3),
#     ('min_rotations',   1),
#     ('max_rotations',   1),
#     ('dn_dlogm', 0),
    
#     ('host_min_mass',   10**12.), # 14 # 13.5
#     ('gal_min_mass',    10**9.5),
#     ('boost_minmass',   10**10.),
    
#     ('host_mass_def',   'm500c'),
#     ('host_R_def',      'r500c'),
#     ('host_id_def',     'CLUID'),
#     ('host_upid_def',   'UID'),

#     ('gal_mass_def',    'm_star'), # 'Macc' #'Vmax_Mpeak' #'M*'), # 'obs_sm') 
#     ('gal_id_def',      'GALID'),
#     ('gal_upid_def',    'UID'),

#     # Magneticum
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.704, # not used
#                  'Omega_m': 0.272,
#                  'Omega_l': 0.728,
#                  'c': 299792.458  # [km/s]
#                  }),
    
#     ('samp_hosts', 1),
#     ('samp_gals',  1),
# ])

# new_config = OrderedDict([
#     # Uchuu with UM!
#     ('catalog_name',    'UchuuUM_sn50_z=0-0.0231_sm9.5-extralarge'),
#     ('host_file',       'um_uchuu/um_uchuu_galdir50_Mhost13.5_galSM9.5.csv'),        
#     ('gal_file',        None),
#     ('z',               0.0231),

#     ('volume', 2000.**3),

#     ('cut_size',        'extralarge'),
    
#     ('host_min_mass',   10**13.9), # 14 # 13.5
#     ('gal_min_mass',    10**9.),
#     ('boost_minmass',   10**14.),
    
#     ('host_mass_def',   'Mnow'),
#     ('host_R_def',      'Rvir'),
#     ('host_id_def',     'id'),
#     ('host_upid_def',   'upid'),

#     ('gal_mass_def',    'Obs_SM'), # 'Macc' #'Vmax_Mpeak' #'M*'), # 'obs_sm') 
#     ('gal_id_def',      'id'),
#     ('gal_upid_def',    'upid'),

#     # Uchuu
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.6774, # not used
#                  'Omega_m': 0.3089,
#                  'Omega_l': 0.6911,
#                  'c': 299792.458  # [km/s]
#                  }),
    
#     ('split_rot', 200),
    
#     ('samp_hosts', 1),
#     ('samp_gals',  1),
# ]) 

import pandas as pd
snap_fname = '/hildafs/home/mho1/hilda/halo_cnn/data_query/snaps.csv'
snapinfo = pd.read_csv(snap_fname).set_index('MDPL2__Redshifts__snapnum')

sni = 121
snapinfo = snapinfo.loc[sni]

new_config = OrderedDict([
    # Uchuu with UM!
    ('catalog_name',    f'MDPL2_UM_sn{sni}_sm9.5_extralarge_031222'),
    ('host_file',       f'cosmosim/MDPL2_Rockstar_snap:{sni}.csv'),
    ('gal_file',        f'um_mdpl2_hearin/sfr_catalog_{snapinfo["MDPL2__Redshifts__aexp"]:.6f}.npy'),
    ('z',               snapinfo['MDPL2__Redshifts__zred']),

    ('volume', 1000.**3),

    ('cut_size',        'extralarge'),
    
    ('host_min_mass',   10**13.5), # 14 # 13.5
    ('gal_min_mass',    10**9.5),
    ('boost_minmass',   10**14.),
    
    ('host_mass_def',   'M200c'),
    ('host_R_def',      'Rvir'),
    ('host_id_def',     'rockstarId'),
    ('host_upid_def',   'upId'),

    ('gal_mass_def',    'obs_sm'), # 'Macc' #'Vmax_Mpeak' #'M*'), # 'obs_sm') 
    ('gal_id_def',      'id'),
    ('gal_upid_def',    'upid'),

    # MDPL2
    ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
                 'h': 0.6777,
                 'Omega_m': 0.307115,
                 'Omega_l': 0.692885,
                 'c': 299792.458  # [km/s]
                 }),
    
    ('split_rot', 10),
    
    ('samp_hosts', 1),
    ('samp_gals',  1),
])

# new_config = OrderedDict([
# #         # UM SM cut with z=0.022->0.0231
#     ('catalog_name',    'MDPL2_UM_z=0.0.022-0231_SM9.5-Xlarge-101321'),
#     ('host_file',       'cosmosim/MDPL2_Rockstar_snap:124.csv'),
#     ('gal_file',        'um_mdpl2_hearin/sfr_catalog_0.978100.npy'),
#     ('z',               0.0231), # coma
    
#     ('dn_dlogm',        10.**-5.2), # -5.2
#     ('dlogm',           0.005),
#     ('boost_minmass',   10**13.),
#     ('min_rotations',   3),
#     ('max_rotations',   500),
    
#     ('volume', 1000.**3),

#     ('cut_size',        'extralarge'),
#     ('split_rot',       10),

#     # MDPL
#     ('cosmo',   {#'H_0': 100.,  # [km/s/(Mpc/h)]
#                  'h': 0.6777,
#                  'Omega_m': 0.307115,
#                  'Omega_l': 0.692885,
#                  'c': 299792.458  # [km/s]
#                  }),
# ])

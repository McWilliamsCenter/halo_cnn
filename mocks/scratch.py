# Scratch file to manipulate/reduce .csv data because it's TOO BIG FOR LOGIN NODE MEMORY

import sys
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tools.matt_tools as matt

wdir = '/home/mho1/scratch/halo_cnn'

MD_file = os.path.join(wdir,
                         'data_query',
                         'MDPL2_Rockstar_snap:120_v2.csv')
#MD_file = os.path.join(wdir,
#                         'data_query',
#                         'MDPL2_Rockstar_z=0.117_Macc=1e11.csv')
UM_file = os.path.join(wdir,
                         'data_raw',
                         'sfr_catalog_0.895100.npy')
print('Loading MD...')
dat_MD = pd.read_csv(MD_file)
print('MD data size: ' + str(sys.getsizeof(dat_MD)/10.**9) + ' GB\n')

print('Loading UM...')
dat_UM = pd.DataFrame(np.load(UM_file))
print('UM data size: ' + str(sys.getsizeof(dat_UM)/10.**9) + ' GB\n')

print('Loading Theory HMF...')
dat_hmf = np.loadtxt(os.path.join(wdir, 'data_raw', 'dn_by_dm_md_om=0.307_s8=0.823_z=0.117.txt'))

x_hmf, y_hmf = zip(*dat_hmf)

print('len MD: ', len(dat_MD))
print('len UM: ', len(dat_UM))


print(dat_MD.head())
print(list(dat_MD.columns))
print(dat_UM.head())
print(list(dat_UM.columns))

print(dat_MD['x'].min(), dat_MD['x'].max())
print(dat_UM['x'].min(), dat_MD['x'].max())

print(dat_MD['Mvir'].min(), dat_MD['Mvir'].max())
print(dat_UM['mvir'].min(), dat_UM['mvir'].max())




print('hosts and kids')

hosts_MD = dat_MD[dat_MD['pId']==-1]
hosts_UM = dat_UM[dat_UM['upid']==-1]
kids_MD = dat_MD[dat_MD['pId']!=-1]
kids_UM = dat_UM[dat_UM['upid']!=-1]

print(len(hosts_MD))
print(len(hosts_UM))

print('above 10**14')
print(np.sum(hosts_MD['Mvir'] > 10**14))
print(np.sum(hosts_UM['mvir'] > 10**14))

print('UM unique hosts: ' + str(len(set(hosts_UM['id']))))
print('UM unique parents: '+ str(len(set(kids_UM['upid']))))

print('plotting hmf')
f = plt.figure(figsize=(7,7))
ax = f.add_subplot(111)

ax.plot(np.log10(x_hmf),y_hmf, label='theo')

matt.histplot(np.log10(hosts_MD['Mvir']),n=50, label='MD',log=1, box=True, ax=ax)
matt.histplot(np.log10(hosts_UM['mvir']),n=50, label='UM',log=1, box=True, ax=ax)
ax.set_xlabel(r'$\log(M_{vir})$', fontsize=20)
ax.set_ylabel(r'dn/dlog(M)', fontsize=20)
ax.legend()

ax.set_title('Hosts HMF Mvir')
f.savefig('/home/mho1/halo_cnn/catalog/HMF_Mvir.pdf')

print('periodic boundary relationships')

## MD
print('~~~ MD ~~~')
low_x_kids = kids_MD[kids_MD['x'] < 1]
high_x_hosts = hosts_MD[hosts_MD['x']>999]

print(len(low_x_kids))
num_high_parents=0
for i, row in low_x_kids.iterrows():
    high_parents= np.sum(high_x_hosts['rockstarId']==row['upId'])
    
    if high_parents >=1:
        num_high_parents+=1
print('MD num_high_parents: ' + str(num_high_parents))

## UM
print('~~~ UM ~~~')
low_x_kids = kids_UM[kids_UM['x'] < 1]
high_x_hosts = hosts_UM[hosts_UM['x']>999]

print(len(low_x_kids))
num_high_parents=0
for i, row in low_x_kids.iterrows():
    high_parents= np.sum(high_x_hosts['id']==row['upid'])
    
    if high_parents >=1:
        num_high_parents+=1
print('UM num_high_parents: ' + str(num_high_parents))
        
print('parent trees')
non_hosts = kids_MD.sample(100)
youngest = 0
num_young = 0
for i, row in non_hosts.iterrows():
    if i%1000==0: print('\n' + str(i))
    
    generations = 1
    parent = dat_MD['pId'][dat_MD['rockstarId']==row['pId']]
    
    if len(parent)==1:
        while parent.iloc[0]!=-1:
            parent = dat_MD['pId'][dat_MD['rockstarId']==parent.iloc[0]]
            if len(parent)>1: print('MULTI')
            generations+=1
    else: print(parent)
    
    youngest = max(youngest, generations)
    if generations>1: 
        num_young+=1
        print(generations)

print('youngest')
print(youngest)
print('num young')
print(num_young)

"""
i_MD = list(hosts_MD['rockstarId'].sort_values())
i_UM = list(hosts_UM['id'].sort_values())

nomatch=0
i_m = 0
for i_u in range(len(i_UM)):
    if i_u%1000==0: print(i_u)
    
    found_match = False
    
    while (i_m < len(i_MD)):
        if i_u%10000==0: print(i_u)
        if i_UM[i_u] == i_MD[i_m]:
            found_match=True
            break
        else:
            i_m += 1
    
    if found_match==False:
        nomatch += 1
        
        i_m = 0
    

print('No match: ',nomatch)
"""

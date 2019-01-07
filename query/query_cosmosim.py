# REFERENCE: vim /home/mho1/.conda/envs/astroq/lib/python2.7/site-packages/astroquery/cosmosim/core.py

import time
import sys
import os
import numpy as np
import datetime as dt
import pandas as pd

from astroquery.cosmosim import CosmoSim

"""
TO-DO: Make this a python module for easier handling. Application in Jupyter notebook.

# General Strategy
1. Submit queries
2. Check for completion, resubmit if error
3. Combine subsets into general catalog


# MDP2 Redshifts

query = "SELECT DISTINCT * FROM MDPL2.Redshifts ORDER BY snapnum DESC"

Snap#:  125     124     123     122     121     120     119     118     117
aexp:   1.0     0.978   0.957   0.936   0.915   0.895   0.8755  0.856   0.838   
z:      0.0     0.022   0.045   0.069   0.093   0.117   0.142   0.168   0.194
"""

save_dir = '/home/mho1/scratch/halo_cnn/data_query'
snapnum = 117
Macc_min = '1e11'
cache = True# Store query locally

# Make subset directory
subset_dir = os.path.join(save_dir, 'subsets')
if not os.path.isdir(os.path.join(subset_dir)):
    os.makedirs(subset_dir)

query_name =  'MDPL2_Rockstar_' + 'snap:' + str(snapnum) + ''

print('\n')
print('~~~~~ Quering ' + query_name + '~~~~~\n')




CS = CosmoSim()
CS.login(username="maho3", password="password")


dx = 200

x_mins = np.arange(0,1000,dx)

table_names = [None]*len(x_mins)
jobids = [None]*len(x_mins)


def submit_query(i, cache=cache):
    xmin = x_mins[i]
    xmax = xmin + dx
    
    # old queries: descId, breadthFirstId, M200b,
    
    sql_query = 'SELECT rockstarId, upId, pId,  Mvir, Rvir, Rs, Macc, Vacc,  M500c, M200c, x, y, z, vx, vy, vz '\
                'FROM MDPL2.Rockstar WHERE snapnum = '+str(snapnum)+\
                ' and Macc > ' + Macc_min + \
                ' and x >=' + str(xmin) + ' and x<' + str(xmax) +''
    
    table_names[i] =    query_name + '_x:[' + str(xmin) + ',' + str(xmax) + ']'
    
    
    print('Querying: ' + table_names[i])

    jobid = CS.run_sql_query(   query_string=sql_query, 
                                tablename=table_names[i] + '_',
                                queue = 'long',
                                cache = cache)
    jobids[i] = jobid
    print('jobid: ' + str(jobid) + '\n')

print('\nSubmitting queries...')
## SUBMIT FIRST QUERIES
for i in range(len(x_mins)):
    submit_query(i)
    time.sleep(1.)
    
# jobids = ['1530314084338316129','1530314089939852270','1530314095439811024','1530314101335877394','1530314106876914667']

    
## CHECK QUERIES, SAVE DATA, RESUBMIT QUERIES AS NECESSARY
t = 0.
completed = [False]*len(x_mins)

print('\nChecking queries...')
while ((np.sum(completed) < len(x_mins)) and (t <= 30)): # set timeout at 30 min

    
    for i in range(len(x_mins)):
    
        if completed[i]: continue
        
        out_file = os.path.join(subset_dir, 
                                table_names[i] + '.csv')
        if os.path.isfile(out_file):
            print('Subset already downloaded.\n')
            completed[i] = True
            continue
                
        time.sleep(1.)
        
        print('Checking status of: ' + table_names[i])
        status = CS.check_job_status(jobid=jobids[i])
        
        print(status)
        
        if status=='COMPLETED':

            CS.download(jobid=jobids[i],
                        filename = out_file,
                        format='csv',
                        cache=cache)
  
            completed[i] = True
                        
        elif status in ['EXECUTING','QUEUED']:
            pass
        elif status in ['ARCHIVED']:
            print('Resubmitting...')
            
            submit_query(i, False)
        else:
            CS.general_job_info(jobid=jobids[i], output=True)
            CS.delete_job(jobid=jobids[i])
            
            print('Resubmitting...')
            
            submit_query(i)
    
    print('Waiting 5sec...')
    time.sleep(5.)
    t += 0.05

if (np.sum(completed) == len(x_mins)):
    print('Successfully executed all data subsets')
else:
    raise Exception('Timeout.')


CS.logout()

    
## COMBINE SUBSETS
print ('\nCOMBINING SUBSETS')

out_dat = None

for i in range(len(x_mins)):

    out_file = os.path.join(subset_dir, table_names[i]  + '.csv')
    
    print('Loading ' + table_names[i] + ' ...')
    
    if i==0:
        out_dat = pd.read_csv(out_file)
    else:
        out_dat = out_dat.append(pd.read_csv(out_file))
        
out_dat = out_dat.reset_index(drop=True)
    
out_file = os.path.join(save_dir,
                        query_name + '.csv')
print('Saving to ' + out_file)
out_dat.to_csv(out_file)



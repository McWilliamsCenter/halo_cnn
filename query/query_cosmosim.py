# REFERENCE: vim /home/mho1/.conda/envs/astroq/lib/python2.7/site-packages/astroquery/cosmosim/core.py

import time
import sys
import os
import numpy as np
import datetime as dt
import pandas as pd

from astroquery.cosmosim import CosmoSim

"""
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
snapnum = 120
Macc_min = '1e11'
cache = True# Store query locally

# Make subset directory
subset_dir = os.path.join(save_dir, 'subsets')
if not os.path.isdir(os.path.join(subset_dir)):
    os.makedirs(subset_dir)


print('\n')
print('~~~~~ Quering MDPL2 at snapnum = ' + str(snapnum) + ' ~~~~~\n')




CS = CosmoSim()
CS.login(username="maho3", password="password")


dx = 250

x_mins = np.arange(0,1000,dx)

table_names = [None]*len(x_mins)
jobids = [None]*len(x_mins)


def submit_query(i):
    xmin = x_mins[i]
    xmax = xmin + dx
    
    sql_query = 'SELECT rockstarId, upId, M200b, Rvir, x, y, z, vx, vy, vz, '\
                'M500c, Rs, Macc, Vacc, Mvir, M200c, scale '\
                'FROM MDPL2.Rockstar WHERE snapnum = '+str(snapnum)+\
                ' and Macc > ' + Macc_min + \
                ' and x >=' + str(xmin) + ' and x<' + str(xmax)
    
    table_names[i] =    'MDPL2_Rockstar_' + 'snap:' + str(snapnum) + \
                        '_x:[' + str(xmin) + ',' + str(xmax) + ']'
    
    
    print('Querying: ' + table_names[i])
    
    jobid = CS.run_sql_query(   query_string=sql_query, 
                                tablename=table_names[i],
                                queue = 'long',
                                cache = cache)
    jobids[i] = jobid
    print('jobid: ' + str(jobid) + '\n')

print('\nSubmitting queries...')
## SUBMIT FIRST QUERIES
for i in range(len(x_mins)):
    submit_query(i)
    time.sleep(1.)
    
    
    
## CHECK QUERIES, SAVE DATA, RESUBMIT QUERIES AS NECESSARY
t = 0.
completed = [False]*len(x_mins)

print('\nChecking queries...')
while ((np.sum(completed) < len(x_mins)) and (t <= 30)): # set timeout at 30 min

    
    for i in range(len(x_mins)):
    
        if completed[i]: continue
        
        print('Checking status of: ' + table_names[i])
        status = CS.check_job_status(jobid=jobids[i])
        
        if status=='COMPLETED':
            out_file = os.path.join(subset_dir, 
                                    table_names[i] + '_' + jobids[i] + '.csv')

            if os.path.isfile(out_file):
                print('Subset already downloaded.')
            else:
                CS.download(jobid=jobids[i],
                            filename = out_file,
                            format='csv',
                            cache=cache)
  
            completed[i] = True
                        
        elif status in ['EXECUTING','QUEUED']:
            print(status + '\n')
            
        else:
            print(status)
            CS.general_job_info(jobid=jobids[i], output=True)
            CS.delete_job(jobid=jobids[i])
            
            print('Resubmitting...')
            
            submit_query(i)
    
    print('Waiting 10sec...')
    time.sleep(10.)
    t += 0.1

if (np.sum(completed) == len(x_mins)):
    print('Successfully executed all data subsets')
else:
    raise Exception('Timeout.')

"""
print('deleting jobs...')
for i in range(len(x_mins)):
    time.sleep(1.)
    CS.delete_job(jobid=jobids[i])"""


CS.logout()

# CS.delete_all_jobs(phase=['ERROR','ABORTED'])

    
## COMBINE SUBSETS
print ('TBD')
"""
out_dat = None

for i in range(len(x_mins)):

    out_file = os.path.join(subset_dir, table_names[i]  + '.csv')
    
    if i==0:
        out_dat = 
"""

# REFERENCE: vim /home/mho1/.conda/envs/astroq/lib/python2.7/site-packages/astroquery/cosmosim/core.py

import time
import sys
import os
import numpy as np
import datetime as dt

from astroquery.cosmosim import CosmoSim

"""
# MDP2 Redshifts

query = "SELECT DISTINCT * FROM MDPL2.Redshifts ORDER BY snapnum DESC"

Snap#:  125     124     123     122     121     120     119     118     117
aexp:   1.0     0.978   0.957   0.936   0.915   0.895   0.8755  0.856   0.838   
z:      0.0     0.022   0.045   0.069   0.093   0.117   0.142   0.168   0.194
"""

save_dir = '/home/mho1/scratch/data_query'
snapnum = 120
Macc_min = '1e11'
cache = False # Store query locally

# Make subset directory
subset_dir = os.path.join(save_dir, 'subsets')
if not os.path.isdir(os.path.join(subset_dir):
    os.makedirs(subset_dir)



print('~~~~~ Quering MDPL2 at snapnum = ' + str(snapnum) + ' ~~~~~\n')




CS = CosmoSim()
CS.login(username="maho3", password="password")


dx = 100

x_mins = np.arange(0,1000,dx)

table_names = [None]*len(x_mins)
jobids = [None]*len(x_mins)


def submit_query(i):
    xmin = x_mins[i]
    xmax = xmin + dx
    
    sql_query = 'SELECT rockstarId, upId, M200b, Rvir, x, y, z, vx, vy, vz, '\
                'M500c, Rs, Macs, Vacc, Mvir, M200c, scale '\
                'FROM MDPL2.Rockstar WHERE snapnum = '+str(snapnum)+\
                ' and Macc > ' + Macc_min + \
                ' and x >=' + str(xmin) + ' and x<' + str(xmax)
    
    table_names[i] = str(dt.datetime.now()) + \
                    '_x = [' + str(xmin) + ', ' + str(xmax) ']'
    
    print('Querying: ' + table_names[i])
    
    jobid = CS.run_sql_query(   query_string=sql_query, 
                                tablename=table_names[i],
                                queue = 'long',
                                cache = cache)
    jobids[i] = jobid
    print('jobid: ' + str(jobid) + '\n')


## SUBMIT FIRST QUERIES
for i in range(len(x_mins)):
    submit_query(i)
    
    
    
## CHECK QUERIES, SAVE DATA, RESUBMIT QUERIES AS NECESSARY
t = 0.
completed = [False]*len(x_mins)

while ((np.sum(completed) < len(x_mins)) and (t <= 30)): # set timeout at 30 min

    print('Waiting 10sec...\n')
    time.sleep(10.)
    t += 0.1
    
    for i in range(len(x_mins)):
    
        if completed[i]: continue
        
        print('Checking status of: ' + table_names[i])
        status = CS.check_job_status(jobid=jobids[i])
        
        if status=='COMPLETED':
            out_file = os.path.join(subset_dir, table_names[i] + '.csv')
        
            CS.download(jobid=jobids[i],
                        filename = out_file,
                        format='csv',
                        cache=cache)
                        
            completed[i] = True
                        
        elif status=='EXECUTING':
            print('still executing...\n')
            
        else:
            print(status)
            print('Resubmitting...')
            
            submit_query(i)

if (np.sum(completed) == len(x_mins)):
    print('Successfully downloaded all data subsets')
else:
    raise Exception('Timeout.')
    
    
## COMBINE SUBSETS



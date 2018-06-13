#Usage: instructions for how to run a bunch of MD queries



#to start an interactive job, 
$ qsub -q physics -l nodes=1:ppn=4 -I -l walltime=48:00:00


#open python
$ python2.7

#imports, etc.
>>> from astroquery.cosmosim import CosmoSim as CS
>>> import time
>>> import sys
>>> import numpy as np
>>> outFolder = '/physics2/cmhicks/Multidark/MDPL2_lightcone/'
>>> CS.login(username = 'ntampaka')

#this will ask for your password



# In theory, the following should let you exit the interactive job and leave it running.  
# In practice, this will submit queries, but will not download data :/
>>> ctrl-z
$ bg
$ disown
$ exit

snapNumList = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]
zList = [0.304, 0.276, 0.248, 0.221, 0.194, 0.168, 0.142, 0.117, 0.093, 0.069,0.045, 0.022, 0.000]
def checkForMissingFiles():
    for i, z in enumerate(zList):
    for xmin in np.arange(0, 1000, dx):
    	fname = outFolder +'z='+str(zList[i])+'_'+str(xmin)+'.csv'
	      if not os.path.isfile(fname):
	      print 'need to make file:', fname
	      xmax = xmin + dx
	      sql_query = 'SELECT rockstarId, upId, M200b, Rvir, x, y, z, vx, vy, vz, M500c, 'Rs, Macc, Vacc, Mvir, M200c, scale FROM MDPL2.Rockstar WHERE snapnum = '+str(snapnum)+' and Macc > 1e11 and x >='+str(xmin)+' and x<'+str(xmax)
	      CS.run_sql_query(query_string=sql_query, queue='long')
	      jobid = CS.current_job
	      CS.download(jobid = str(jobid),  filename = outFile, format = 'csv')

    

def runAllJobs():
    snapNumList = [125, 117, 124]
    zList = [0.000, 0.194, 0.022]
    dx = 100
    jobList = []
    fileNameList = []
    for i, snapnum in enumerate(snapNumList):
        for xmin in np.arange(0, 1000, dx):
            xmax = xmin + dx
            sql_query = 'SELECT rockstarId, upId, M200b, Rvir, x, y, z, vx, vy, vz, M500c, '\
                    'Rs, Macs, Vacc, Mvir, M200c, scale '\
                    'FROM MDPL2.Rockstar WHERE snapnum = '+str(snapnum)+\
                    ' and Macc > 1e11 and x >='+str(xmin)+' and x<'+str(xmax)
            print ''
            print ''
            print 'snapnum = ', snapnum
            print 'query   = ', sql_query
            print 'xrange:', xmin, xmax
            CS.run_sql_query(query_string=sql_query, queue='long')
            jobid = CS.current_job
            print 'jobid   =', jobid
            jobList.append(jobid)
            print 'jobList:', jobList
            outFile = outFolder +'z='+str(zList[i])+'_'+str(xmin)+'.csv'
            fileNameList.append(outFile)
            print 'fileNameList:', fileNameList
            downloaded = False
            counter = 0
            while downloaded == False:
                if 'EXECUTING' in CS.check_job_status(str(jobid)):
                    time.sleep(600.)  #wait for 10 mins
                    counter += 1
                    if counter >7:
                        print 'waited too ong on job.'
                        print 'jobid:', jobid
                        print 'file:', outFile
                        print ''
                        downoaded = True
                elif 'ABORTED' in CS.check_job_status(str(jobid)):
                    print 'job was aborted.'
                    print 'jobid:', jobid
                    print 'file:', outFile
                    print ''
                    downoaded = True
                else:
                    CS.download(jobid = str(jobid),  filename = outFile, format = 'csv')
                    downloaded = True
    print 'joblist:', jobList
    print 'filenameList:', fileNameList


if __name__ == "__main__":
    main()

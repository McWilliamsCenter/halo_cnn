#!/usr/bin/env python2.7

#Usage: apply SDM (Sutherland 2012) to HeCS catalog


import numpy as np
from array import array
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
import sys
import sdm
import argparse
import pickle
from sklearn.cross_validation import KFold
from scipy.stats import gaussian_kde
from os import listdir

import pandas as pd

import os

from collections import OrderedDict
import time


import tools.matt_tools as matt
from tools.catalog import Catalog


matplotlib.rc('text',usetex=True)
matplotlib.rc('font',family='serif')
matplotlib.rc('lines',linewidth=3)
fontSize = 20
fontColor = 'k'

#############################################################################
pi = math.pi


###################### PARAMETERS: ###########################################################
redshift = '0.117'
# trainFile = '/share/scratch1/cmhicks/mattho/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift +'.npy' #training file
# trainFile = '/share/scratch1/cmhicks/mattho/debugcat.npy'
#trainFile = '/home/mho1/data/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift + '_vfiltscale.npy'
trainFile = '/home/mho1/scratch/halo_cnn/data_mocks/Rockstar_UM_z=0.117_contam_med.p'
subsample = 0.1

scenario = 'sdm_mich' #name this run
featureType = 'MLv' #MLv, MLR, MLvR, MLvsR ... tells which features we're going to use
Nfolds = 10
divfunc = 'kl'
K = 4
sdrModel = 'NuSDR' #NuSDR, SDR

n_proc = 28 #set the number of processors to run on
seed = 1111209 #favorite integer
stand = True #standardizes features

outFolder = '/home/mho1/scratch/halo_cnn/mich/' #where to put the output
# logFolder = 'log/'

#if no input file of unlabeled data, leave as None
unlabeledFile = None #'/home/cmhicks/VDF_HeCS/HeCS/output/HeCS_pared.npy'


par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_sdm1d'),

    ('in_folder'    ,   'data_mocks'),
    ('data_file'    ,   'Rockstar_UM_z=0.117_contam_med.p'),

    ('subsample'    ,   0.1 ),
    ('nfolds'       ,   10),
    
    ('test_range'   ,   (10**13.9, 10**15.1)),
    
    ('dn_dlogm'     ,   10.**-5.2),
    ('dlogm'        ,   0.02),


    ('sdrModel'     ,   'NuSDR'),
    ('K'            ,   4),
    ('divfunc'      ,   'kl'),
    ('progressbar'  ,   True)

])

def set_params(config):
    
    for param in config.keys():
        if param not in globals().keys():
            continue
            
        execstr = "global "+param+"; "+param + ' = '
             
        if type(config[param])==str:
            execstr += '"' + config[param] + '"'
        else: execstr += str(config[param])
        
        exec(execstr)

    print ('~~~ CONFIGS ~~~')
    print (config)
    
    return 

def loadHalos(halofile):
    ## DATA
    print('\n~~~~~ LOADING DATA ~~~~~')
    # Load and organize

    in_path = os.path.join(par['wdir'], par['in_folder'], par['data_file'])

    cat = Catalog().load(in_path)

    if (cat.par['vcut'] is None) & (cat.par['aperture'] is None):
        print('Pure catalog...')
        cat.par['vcut'] = 3785.
        cat.par['aperture'] = 2.3





    print('\n~~~~~ DATA CHARACTERISTICS ~~~~~')
    for key in cat.par.keys():
        print(key,':', cat.par[key])

    print('\n~~~~~ ASSIGNING TEST/TRAIN ~~~~~')

    in_train = np.array([False]*len(cat))
    in_test = np.array([False]*len(cat))

    log_m = np.log10(cat.prop['M200c'])
    bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['dlogm'])*1.0001, par['dlogm'])
    n_per_bin = int(par['dn_dlogm']*1000**3*par['dlogm'])

    for j in range(len(bin_edges)):
        bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['dlogm']) ].values
        
        if len(bin_ind) <= n_per_bin:
            in_train[bin_ind] = True # Assign train members
        else:
            in_train[np.random.choice(bin_ind, n_per_bin, replace=False)] = True

    in_test[cat.prop.index[(cat.prop['rotation'] < 3) & 
            (cat.prop['M200c'] > par['test_range'][0]) &
            (cat.prop['M200c'] < par['test_range'][1])].values] = True


    print('\n~~~~~ REMOVING UNUSED DATA ~~~~~')
    keep = (in_train + in_test) > 0

    cat = cat[keep]
    in_train = in_train[keep]
    in_test = in_test[keep]


    print('\n~~~~~ ASSIGNING FOLDS ~~~~~~')
    # Use rank-ordering to assign folds evenly for all masses

    ids_massSorted = cat.prop[['rockstarId','M200c']].drop_duplicates().sort_values(['M200c','rockstarId'])['rockstarId']

    fold_ind = pd.Series(np.arange(len(ids_massSorted)) % par['nfolds'], 
                         index = ids_massSorted)

    fold = fold_ind[cat.prop['rockstarId']].values

    for i in range(par['nfolds']):
        print('Fold #' + str(i) + ' --> train:' + str(np.sum(in_train[fold!=i])) + ' test:' + str(np.sum(in_test[fold==i])) )

    # Subsample
    print('\n~~~~~ SUBSAMPLING ~~~~~')
        
    print('Data length: ' + str(len(cat)))

    if par['subsample'] < 1:
        ind = np.random.choice(range(len(cat)), 
                               int(par['subsample']*len(cat)),
                               replace=False
                              )
        cat = cat[ind]
        in_train = in_train[ind]
        in_test = in_test[ind]
        fold = fold[ind]

    print('Subsampled data length: ' + str(len(cat)))

    return (cat, in_train, in_test, fold)



def setSDMModel(sdrModel, n_proc, divfunc, K):
    if sdrModel == 'NuSDR':
        print('NuSDR')
        model = sdm.NuSDR(n_proc = n_proc, div_func = divfunc, K=K, progressbar=True)
    elif sdrModel == 'SDR':
        model = sdm.SDR(n_proc = n_proc, div_func = divfunc, K=K, progressbar=True)
    return model


def makeFeaturesList(halos, foldList, catname, featureType, \
                         stand=True, scaler = None):

    cat, in_train, in_test, fold = halos
    
    if foldList == None:

        print('check:  this should be used for unlabeled data and its training sample!')
        #we're passed a file, but no foldList, so we are going to use all of it
        if catname == 'train':
            ind = np.argwhere(halos['intrain'] == 1)
        elif catname == 'test':
            ind = np.arange(len(halos))

    else:
        print('check:  this should be used for labeled data!')
        #we're passed a big file and have to figure out what is in the train and test catalog
        inFold = [True if (fold[i] in foldList) else False for i in range(len(cat))]
        if catname == 'train':
            ind = np.argwhere(inFold & in_train == 1)
            print(' --> train:' + str(len(ind)))
        elif catname == 'test':
            ind = np.argwhere(inFold & in_test == 1)
            print(' --> test:' + str(len(ind)))



    numHalos = len(ind)

    """
    if catname == 'train':
        #whatever goes here is the thing we're predicting... in this case, log(M500):
        massList = np.ndarray.flatten(np.log10(halos['Mtot'][ind])).tolist()
    else:
        massList = [0]*len(ind)
    """

    massList = np.ndarray.flatten(np.log10(cat.prop['M200c'].values[ind])).tolist()
    
    print('massList check:', massList[0:10])
    IDList = [i for i in fold[ind].flatten()]
    fold = np.ndarray.flatten(fold[ind]).tolist()

    print('number of objects in ', catname, 'catalog with fold(s) = ', foldList, ':  ', len(ind))


    #note:  my old and new catalogs take slightly different forms.  To get everything to talk,
    #       I had to insert a *ton* of ".flatten()"

    if featureType == 'MLv':
        print('MLv')
        #uses |v| as the only featuer
        featuresList = [[(j,1) for j in np.abs(cat.gal[m]['vlos'])] for m in ind.flatten()]

    elif featureType == 'MLvR':
        print('MLvR')
        #uses 2 features:  |v|, R
        featuresList = [list(zip(np.abs(cat.gal[i]['vlos']), 
                        cat.gal[i]['Rproj'])) for i in ind.flatten()]
            
    feats = sdm.Features(featuresList, mass = massList, default_category = catname)
   


    if stand == True or stand == 'True' or stand == 'true':
        #standardize the features

        #if this is the first time through, choose a scaler:
        if scaler == None:
            print('applying standardization;',)
            feats, scaler = feats.standardize(ret_scaler=True)
            print('scaler = ', scaler.mean_)
        
        #otherwise, use the scaler that was passed:
        else:
            print('standardizing features with scaler = ', scaler.mean_)
            feats = feats.standardize(scaler = scaler)

    return feats, massList, IDList, scaler, numHalos
    


def predictLabeledData(n, halos):

    print('=================  Fold', n, '===================')

    # create train data with all but one of the folds:
    trainFoldList = list(range(Nfolds))
    trainFoldList.remove(n)
    traindata, trainmass, foldIDList, scaler, junk = \
                makeFeaturesList(halos, trainFoldList, 'train', featureType, stand, \
                                     scaler = None)
    
    # and create test data with the fold of interest:
    testdata, testmass, junk, junk, numHalos = \
         makeFeaturesList(halos, [n], 'test', featureType, stand, scaler)

    #make an array to hold output:
    #0 = predicted mass
    #1 = true mass
    #2 = fold
    preds = np.zeros((len(testmass), 3), 'f')

    
    # assign the training folds (NOTE: THIS WILL NOT WORK FOR ARBITRARILY SMALL NFOLDS, 
    # AND MIGHT DO SOMETHING REALLY DUMB FOR SOME NFOLDS VALUES
    # THIS WILL WORK BEST IF NFOLDS = 3*N+1, WHERE N IS SOME INTEGER.  
    # SO FOR 10 FOLDS, THIS WORKS NICELY, BUT PROCEED WITH CAUTION OTHERWISE
    print('attempting to assign crossvalidation folds:')
    halo_folds = KFold(n=len(set(foldIDList)), n_folds=3, shuffle=True)
    model._tune_folds = [[np.vectorize(x.__contains__)(foldIDList) for x in traintest] \
                             for traintest in halo_folds]


    #a debugging relic:
    #print 'checking isfinite:'
    #for item in [trainmass, testmass, traindata.mass]:
    #    print np.all(np.isfinite(item)),  np.min(item), np.max(item)
    #    print len(set(item)), len(item)
    #    print ''

    print('traindata:', traindata)
    print('testdata:', testdata)
    print('model:', model)
    #crossvalidate:
    print('crossvalidating')
    preds[:, 0] = \
            model.transduct(traindata, traindata.mass, testdata, save_fit=True)

    # put the true mass into a file:
    preds[: , 1] = testmass
    # as well as the fold
    preds[: , 2] = n

    # print out some tuning parameters:
    print('tuning parameters:', )
    print(model._tuned_params())
    
    return preds


def run():
    
    # set up a timer
    t0 = time.clock()
    start_t = time.time()

    
    #count multiple runs
    # log = listdir(logFolder)
    # num = int(np.sum([(i[:(5+len(scenario))] == ('log_' + scenario + '_')) for i in log]))
    
    # print("log folder at: " + logFolder + 'log_' + scenario+'_' + str(num) + '.txt')
    
    #ouptut progress to a file
    # sys.stdout = open(logFolder + 'log_' + scenario+'_' + str(num) + '.txt', 'w', 1)

    print("~~~ start: %s ~~~" % str(start_t))

    #Messrs. Moony, Wormtail, Padfoot, and Prongs
    #Purveyors of Aids to Magical Mischief-Makers
    #are proud to present SDM predictions

    #random seed:
    np.random.seed(seed)

    #get SDM ready to roll:
    global model
    model = setSDMModel(sdrModel, n_proc, divfunc, K)

    #load the halos:
    halos = loadHalos(trainFile)
    

    #predict labeled data for each of the folds:
    preds = np.zeros((0,3), 'f')
    for n in range(Nfolds):
        preds = np.append(preds, predictLabeledData(n, halos), axis=0)

    print(preds.shape)

    save_dict = {
        'logmass_test'    :   preds[:,1],
        'logmass_pred'    :   preds[:,0],
        'fold'            :   preds[:,2]
    }

    # np.save(os.path.join(model_dir, model_name_save + '.npy'), save_dict)
    
    np.save(outFolder+scenario+'_preds.npy', save_dict)

    print("~~~ %s hours ~~~" % ((time.time() - start_t) / (60*60) ))
        
    return 0

run()

"""
Excess code

    global redshift, trainFile, scenario, featureType, Nfolds, K, sdrModel, n_proc, seed, stand, outFolder, logFolder, unlabeledFile
    
    redshift = config['redshift']
    trainFile = config['trainFile']
    scenario = config['scenario']
    featureType = config['featureType']
    Nfolds = config['Nfolds']
    K = config['K']
    sdrModel = config['sdrModel']
    n_proc = config['n_proc']
    seed = config['seed']
    stand = config['stand']
    outFolder = config['outFolder']
    logFolder = config['logFolder']
    unlabeledFile = config['unlabeledFile']
    subsample = config['subsample']

"""

#these are the detectors we will use
from detectors.expose.expose_detector import ExposeDetector
from detectors.condexpose.condexpose_detector import CondexposeDetector
from detectors.null.null_detector import NullDetector
from detectors.spirit.spirit_detector import SpiritDetector


#these files contain helper functions
from preprocessor import Preprocessor
from runner import Runner
from adPlot import AdPlot

#NAB files used
from nab.util import (absoluteFilePaths,
                      createPath,
                      byteify,
                      getGridParams)
from nab.corpus import Corpus, DataFile

#general libraries
import pandas as pd
import os
import json
import random
#%%

#set up paths and folders
root = os.path.abspath('..')
rawDir = os.path.join(root,'data_raw')
dataDir = os.path.join(root,'data_preprocessed')
labelDir = os.path.join(root,'labels')
print("root     : " + root)
print("raws     : " + rawDir)
print("dataDir  : " + dataDir)
print("labelDir : " + labelDir)
#%%
print('\n ==Preprocessing==========================================')
#construct a dict per collection of datasets
#the idea is that each collection has datasets with similar formats
collections = {
    #"nab" : {'folder':'/nab/','delimiter':',','timestamp':True},
    "synth" : {'folder':'/synth/','delimiter':',','timestamp':False}
}

#now use the collections to create preprocessors and automatically preprocess each folder
for c in collections:
    rawPaths = absoluteFilePaths(rawDir+collections[c]['folder'])
    for f in rawPaths:
        print('\n **Preprocessing ' + f)
        pre = Preprocessor(fileName=f,
                           delimiter=collections[c]['delimiter'],
                           timestamp=collections[c]['timestamp'],
                           folderStruct=collections[c]['folder'])
        pre.autoPP()
print('\n ==Finished Preprocessing==================================')

#create a dict via nab.corpus that contain filenames as keys and dataframes as values
print('\n Loading preprocessed data files to corpus')
corpus = Corpus(dataDir)
data = corpus.dataFiles
print('The following datasets will be analysed')
for d in data:
    print(d)
print('\n Finished loading preprocessed data files to corpus')

#load labels
print('\n Loading labels.json')
with open('/home/raph/ad-thesis/labels/combined_labels.json') as json_file:  
    labels = byteify(json.load(json_file))
print('\n Finished loading labels.json')

detectors = {}
detectorArgs = {}
 
#detectors, detectorArgs = getGridParams(detectors,detectorArgs,'expose',ExposeDetector,['fourierFeatures'],[4000,6000,8000])
#detectors, detectorArgs = getGridParams(detectors,detectorArgs,'expose',ExposeDetector,['decay'],[0.01,0.05,0.1,0.2])
#detectors, detectorArgs = getGridParams(detectors,detectorArgs,'expose',ExposeDetector,['gamma'],[0.1,0.25,0.5,1.0,2.0])

#detectors,detectorArgs = getGridParams(detectors,detectorArgs,'cexpose',CondexposeDetector,['gamma'],[0.1,0.25,0.5,1.0,2.0])
#detectors,detectorArgs = getGridParams(detectors,detectorArgs,'cexpose',CondexposeDetector,['fourierFeatures'],[50,100,150,200,250,300])
#
#detectors,detectorArgs = getGridParams(detectors,detectorArgs,'spirit',SpiritDetector,['lam'],[0.90,0.92,0.94,0.96])
#detectors,detectorArgs = getGridParams(detectors,detectorArgs,'spirit',SpiritDetector,['fE'],[0.90,0.92,0.94,0.96])
#detectors,detectorArgs = getGridParams(detectors,detectorArgs,'spirit',SpiritDetector,['FE'],[0.965,0.97,0.98,0.99])

detectors,detectorArgs = getGridParams(detectors,detectorArgs,'expose',ExposeDetector)
detectors,detectorArgs = getGridParams(detectors,detectorArgs,'cxpose',CondexposeDetector)
detectors,detectorArgs = getGridParams(detectors,detectorArgs,'spirit',SpiritDetector)
#%%
print('\n ==Running detectors on datasets ============================')

plotFileName = "%06x" % random.randint(0, 0xFFFFFF)
runner = Runner(dataSets=data, detectors=detectors, detectorArgs=detectorArgs, plotFileName=plotFileName)
runner.run()
print('\n ==Finished running detectors on datasets ===================')
#%%
print('\n ==Loading results ==========================================')
#load results info
f=open('../results/plotFiles/'+plotFileName, "r")
content = f.readlines()
f.close()
#%%
#turn content to dict
results = {line.split(',')[0]:line[len(line.split(',')[0])+1:-1] for line in content}
dataUrls = [results[i].split(',')[-2] for i in results]
resultUrls = [results[i].split(',')[-1] for i in results]
print('\n ==Finished loading results =================================')

print('\n ==Plotting =================================================')
for i in range(len(resultUrls)):
    plotter = AdPlot(resultsUrl=resultUrls[i], labels=labels[dataUrls[i]])
    plotter.interpret()
    plotter.addToScoreFile(plotFileName)
    plotter.plot(plotFileName, savePlot=True)
    plotter.plotPRCurve(plotFileName)
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
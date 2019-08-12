#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:56:43 2019

Original made by Numenta.
Modified by Raphael Vorias
"""

import multiprocessing
from detectors.base import detectDataSet
from nab.optimizer import optimizeThreshold
from nab.util import updateThresholds
from nab.corpus import Corpus
from nab.scorer import scoreCorpus
from nab.labeler import CorpusLabel
import os
import json

class Runner(object):
    def __init__(self, dataSets, detectors, detectorArgs, plotFileName):
        # detectors do not score during probationary period
        self.probationaryPercent = 0.15
        # defined if needed
        self.windowSize = 0.10
        self.dataSets = dataSets
        self.detectors = detectors
        self.detectorArgs = detectorArgs
        self.outputPaths = []
        self.numCPUs = 3
        self.pool = multiprocessing.Pool(self.numCPUs)
        self.plotFileName = plotFileName
        self.runInParallel = True
        self.resultsDir = 'results'
        
    def run(self):
        '''
        Construct the correct argument per detector per dataset.
        If runInParallel is enabled, then multiprocessing is used.
        '''
        count = 0
        args = []
        for detectorName, detectorConstructor in self.detectors.items():
            for dataSetName, dataSet in self.dataSets.items():
                args.append([count,
                             detectorName,
                             detectorConstructor(dataSet=dataSet.data,
                                          probationaryPercent=self.probationaryPercent),
                             self.detectorArgs[detectorName],
                             dataSetName,
                             self.plotFileName])
                if self.runInParallel == False:
                    detectDataSet(args[-1])
                count += 1
        # Using `map_async` instead of `map` so interrupts are properly handled.
        # See: http://stackoverflow.com/a/1408476
        if self.runInParallel == True:
            self.pool.map_async(detectDataSet, args).get(99999999)
    
    def optimize(self, detectorNames):
        """Optimize the threshold for each combination of detector and profile.
        @param detectorNames  (list)  List of detector names.
        @return thresholds    (dict)  Dictionary of dictionaries with detector names
                                      then profile names as keys followed by another
                                      dictionary containing the score and the
                                      threshold used to obtained that score.
        """
        print("\nRunning optimize step")
        
        thresholds = {}
        
        for detectorName in detectorNames:
          resultsDetectorDir = os.path.join(self.resultsDir, detectorName)
          resultsCorpus = Corpus(resultsDetectorDir)
        
          thresholds[detectorName] = {}
        
          for profileName, profile in self.profiles.iteritems():
            thresholds[detectorName][profileName] = optimizeThreshold(
              (detectorName,
               profile["CostMatrix"],
               resultsCorpus,
               self.corpusLabel,
               self.probationaryPercent))
        
        updateThresholds(thresholds, self.thresholdPath)
        
        return thresholds    
        
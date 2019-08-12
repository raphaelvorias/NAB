import pandas as pd
#import plotly.offline as offline
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from funcsigs import signature

class AdPlot(object):
    def __init__(self, resultsUrl=None, labels=None):
        self.resultsUrl = resultsUrl
        self.labels = labels
        self.loadUrls()
        self.detectorName = resultsUrl.split('/')[2]
        self.dataSetName = resultsUrl.split('/')[-1]
        self.aggregate = {}
        
        self.probation = 0.15
        
    def loadUrls(self):
        '''
        Loads url into a dataframe
        '''
        if self.resultsUrl is not None:
            self.results = pd.read_csv(self.resultsUrl)
            self.dfLength = self.results.shape[0]
            self.dfCols = self.results.shape[1]
    
    def interpret(self):
        '''
        Calculates metrics and shows the result
        @todo: anomalies according to gaussian -> reason if outlier
        @todo: implement sigmoid weighted window scoring
        '''
        
        anomalyWindowPercent = 0.10
        numberOfLabeledAnomalies = len(self.labels)
        self.probationPeriod = int(self.probation * self.dfLength)
        
        mu = self.results['anomalyScore'][self.probationPeriod:-1].mean()
        sig = self.results['anomalyScore'][self.probationPeriod:-1].std()

        #Anomaly window defined like Numenta
        if numberOfLabeledAnomalies > 0:
            anomalyWindowSize = self.dfLength * anomalyWindowPercent / numberOfLabeledAnomalies
        #print('anomalyWindowSize is {}'.format(anomalyWindowSize))

        print('\n ** Results from ' + self.resultsUrl)
        
        self.labelPoints = []
        self.windowLeft = []
        self.windowRight = []
        
        # discrimination function
        # it's an extra step, normally the detector already has a similar filter function
        # to calculate its raw anomaly score
        self.threshold = (mu + 3*sig)

        self.results['trueAnomaly'] = 0.0
        self.results['detectedAnomalies'] = np.where(self.results['anomalyScore'] > self.threshold, 1, 0)
        self.results['windowZone'] = 0
        self.results['window'] = -1.0
        self.results['ignore'] = 0
        
        for i, row in self.results.iterrows():
            if i < self.probationPeriod:
                self.results.at[i,'ignore'] += 1
            if row['timestamp'] in self.labels:
                self.results.at[i, 'trueAnomaly'] =  1
                self.labelPoints.append(i)
                self.windowLeft.append(max(int(i-anomalyWindowSize/2),0))
                self.windowRight.append(min(int(i+anomalyWindowSize/2),self.dfLength))
        
        #see profiles.json in original NAB/config folder
        tpWeight = 1.0
        fnWeight = -1.0
        fpWeight = -1.0
        tnWeight = 1.0
        
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        
        sigmas = []
        
        #Setting the window weights
        for i in range(len(self.windowLeft)):
            for j in range(self.windowLeft[i],self.windowRight[i]+int(anomalyWindowSize/3)):
                y = (j - self.windowRight[i])
                sigma = (tpWeight-fpWeight)*(1.0/(1.0+np.exp(0.04*y)))-1.0
                sigmas.append(sigma)
                self.results.at[j,'window'] = sigma
                self.results.at[j,'windowZone'] = 1
        
        anomalyDetectedInWindow = False
        for i in range(len(self.windowLeft)):
            for j in range(self.windowLeft[i],self.windowRight[i]+int(anomalyWindowSize/3)):
                if self.results.at[j,'detectedAnomalies'] == 1:
                    if anomalyDetectedInWindow == False:
                        self.TP += 1
                        self.results.at[j,'ignore'] -= 1
                        anomalyDetectedInWindow = True
                if anomalyDetectedInWindow == True:
                    self.results.at[j,'ignore'] += 1
            if anomalyDetectedInWindow == False:
                self.FN += 1
            else:
                anomalyDetectedInWindow = False
                
        self.results['TN'] = (self.results['windowZone']-1)*(self.results['detectedAnomalies']-1)*(1-self.results['ignore'])
        self.TN = self.results['TN'].sum()
        
        self.results['FP'] = self.results['detectedAnomalies'] - self.results['windowZone']
        self.results['FP'] = np.where(self.results['FP'] > 0, 1, 0)*(1-self.results['ignore'])
        self.FP = self.results['FP'].sum()
        
        self.results['weighted'] = self.results['window']*self.results['detectedAnomalies']*(1-self.results['ignore'])
        self.nabScore = self.results['weighted'].sum() + fnWeight*self.FN
        
        try:
            precision = float(self.TP)/(self.TP+self.FP)
            recall = float(self.TP)/(self.TP+self.FN)
            F1 = 2 * precision * recall / (precision + recall)
        except:
            print('~!~ Error during calculation of precision and recall: division by zero')
            precision = 0.0
            recall = 0.0
            F1 = 0.0
            
        print('TP :{}'.format(self.TP))
        print('FP :{}'.format(self.FP))
        print('FN :{}'.format(self.FN))
        print('TN :{}'.format(self.TN))
        
        y_score = self.results['anomalyScore'].values[self.probationPeriod:]
        y_test = self.results['trueAnomaly'].values[self.probationPeriod:]
        
        self.average_precision = average_precision_score(y_test, y_score)

        #put values into aggregate
        self.aggregate = {
            'TP':self.TP,
            'FP':self.FP,
            'FN':self.FN,
            'TN':self.TN,
            'precision':precision,
            'recall':recall,
            'F1':F1,
            'nab':self.nabScore,
            'detectorName':self.detectorName,
            'dataSetName':self.dataSetName, #
            'avgPrec':self.average_precision
        }
        

                
    def plotPRCurve(self, plotFileName,savePlot=True):
        print('Average precision-recall score: {0:0.2f}'.format(self.average_precision))
        
        y_score = self.results['anomalyScore'].values[self.probationPeriod:]
        y_test = self.results['trueAnomaly'].values[self.probationPeriod:]
        
        if savePlot is not None and plotFileName is not None:
            try:
                if not os.path.exists('../results/PRcurves/'+plotFileName):
                    os.makedirs('../results/PRcurves/'+plotFileName)
            except OSError:
                print ('Error: Creating directory. ' +  plotFileName)
                
        curvePrec, curveRecall, thresholds = precision_recall_curve(y_test, y_score)
        
        print(thresholds)
        
        step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
        plt.step(curveRecall, curvePrec, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(curveRecall, curvePrec, alpha=0.2, color='b', **step_kwargs)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(self.average_precision))
      
        
        plt.savefig('../results/PRcurves/'+plotFileName+'/'+self.detectorName+'_'+self.dataSetName+'.png',
                    bbox_inches='tight')
        plt.show()
        
    def addToScoreFile(self,plotFileName):
        '''
        Adds results to plotFileName for later retrieval
        '''
        text = ''
        text += str(self.aggregate['detectorName']) + ','
        text += str(self.aggregate['dataSetName']) + ','
        text += str(self.aggregate['TP']) + ','
        text += str(self.aggregate['FP']) + ','
        text += str(self.aggregate['FN']) + ','
        text += str(self.aggregate['TN']) + ','
        text += '{:04.2f}'.format(self.aggregate['precision']) + ','
        text += '{:04.2f}'.format(self.aggregate['recall']) + ','
        text += '{:04.2f}'.format(self.aggregate['F1']) + ','
        text += '{:04.2f}'.format(self.aggregate['nab']) + ','
        text += '{:04.2f}'.format(self.aggregate['avgPrec']) + '\n' #
        
        try:
            if not os.path.exists('../results/scores/'):
                os.makedirs('../results/scores/')
        except OSError:
            print ('Error: Creating score directory.')
        
        f=open('../results/scores/'+plotFileName, "a+")
        f.write(text)
        f.close()
        

    def getAggregateText(self):
        '''
        Generates results in reading format
        '''
        text = ''
        text += 'detectorName: ' + str(self.aggregate['detectorName']) + '\n'
        text += 'dataSetName: ' + str(self.aggregate['dataSetName']) + '\n'
        text += 'TP: ' + str(self.aggregate['TP']) + '    '
        text += 'FP: ' + str(self.aggregate['FP']) + '    '
        text += 'FN: ' + str(self.aggregate['FN']) + '    '
        text += 'TN: ' + str(self.aggregate['TN']) + '    '
        text += 'precision: ' + '{:04.2f}'.format(self.aggregate['precision']) + '    '
        text += 'recall: ' + '{:04.2f}'.format(self.aggregate['recall']) + '    '
        text += 'F1: ' + '{:04.2f}'.format(self.aggregate['F1']) + '\n'

        return text

    def plot(self, plotFileName=None, savePlot=None):
        '''
        Plots the current results
        Optionally save the plot as .png
        '''
        #plt plot
        #legend definitions
        anomaly_patch = mpatches.Patch(color='green', alpha=.2, label='anomaly windows')
        signal_line = mlines.Line2D([], [], color='blue', label='sensor value')
        probation_period = mpatches.Patch(color='red', alpha=.2, label='probation period')
        raw_line = mlines.Line2D([], [], color='blue', label='raw score')
        detected_line = mlines.Line2D([], [], color='blue', label='detected anomaly')
        
        
        f, axarr = plt.subplots(self.dfCols, 1, sharex=True, figsize=(12, self.dfCols+1), dpi=80)
        plt.subplots_adjust(hspace=0.6)
        f.suptitle(self.getAggregateText(), fontsize=12)  

        for i in range(self.dfCols-2):
            axarr[i].plot(self.results['v_{}'.format(i)],label='signal')
            axarr[i].set_title('v_{}'.format(i))
            
        axarr[0].legend(handles=[probation_period,anomaly_patch,signal_line],
                 loc='upper left',
                 bbox_to_anchor=(0.85, 3.0),
                 ncol=1,
                 fancybox=True,
                 shadow=False)

        axarr[self.dfCols-2].plot(self.results['anomalyScore'],label='raw score')
        axarr[self.dfCols-2].set_title('raw anomaly score')
        axarr[self.dfCols-2].set_ylim([-.1, 1.1])
        #axarr[self.dfCols-2].legend(handles=[anomaly_patch, raw_line],loc='upper left')

        axarr[self.dfCols-1].plot(self.results['detectedAnomalies'],label='filtered anomalies')
        axarr[self.dfCols-1].set_title('filtered anomalies (three sigma)')
        axarr[self.dfCols-1].set_ylim([-.1, 1.1])
        #axarr[self.dfCols-1].legend(handles=[anomaly_patch, detected_line],loc='upper left')

        #anomaly windows
        for i in range(self.dfCols):
            for j in range(len(self.labels)):
                axarr[i].axvspan(self.windowLeft[j], 
                    self.windowRight[j], 
                    facecolor='g', 
                    edgecolor='none', 
                    alpha=.2,
                    label='window')
        
        #probation period        
        for i in range(self.dfCols):
            axarr[i].axvspan(0, 
                int(self.probation * self.dfLength), 
                facecolor='r', 
                edgecolor='none', 
                alpha=.1,
                label='probation period')
                
        plt.show()
        if savePlot is not None and plotFileName is not None:
            
            try:
                if not os.path.exists('../results/plotImages/'+plotFileName):
                    os.makedirs('../results/plotImages/'+plotFileName)
            except OSError:
                print ('Error: Creating directory. ' +  plotFileName)
            
            #f.savefig('../results/plotImages/'+self.detectorName+'_'+self.dataSetName+'.svg')
            f.savefig('../results/plotImages/'+plotFileName+'/'+self.detectorName+'_'+self.dataSetName+'.png',
                      bbox_inches='tight')
#%%
#Test
#plotter = AdPlot(resultsUrl='../results/cexpose_gamma_1.0/synth/baselineSinesMixed_2019-08-01_20-46-33.csv',
#                 labels=[
#  	"2010-01-12 00:33:20",
#  	"2010-01-12 00:43:20",
#  	"2010-01-12 00:53:20",
#  	"2010-01-12 01:06:40"
#  ])
#plotter.interpret()
#plotter.addToScoreFile('auc_test')
#plotter.plot('auc_test', savePlot=True)
#plotter.plotPRCurve('auc_test')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
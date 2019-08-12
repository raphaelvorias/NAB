import numpy as np
from numpy import linalg as LA
from scipy import stats

from detectors.base import AnomalyDetector

def mm(a,b):
    return np.matmul(a,b)
def inv(a):
    return np.linalg.inv(a)

class SpiritDetector(AnomalyDetector):
	'''
	Implementation of SPIRIT, original paper:
	Papadimitriou, Spiros, Jimeng Sun, and Christos Faloutsos. 
	“Streaming Pattern Discovery in Multiple Time-Series,” 2005, 12.
	'''

  def __init__(self, *args, **kwargs):
    super(SpiritDetector, self).__init__(*args, **kwargs)

  def initialize(self,lam=None,fE=None,FE=None):
    """
    
    """
    self.n = self.cols-1
    #decay factor
    if lam is None:
        self.lam = 0.90
    else:
        self.lam = lam
        
    if fE is None:
        self.fE = 0.90
    else:
        self.fE = fE
        
    if FE is None:
        self.FE = 0.96
    else:
        self.FE = FE
        
    print('parameters -- lambda={} fE={} FE={}'.format(self.lam, self.fE, self.FE))
    
    self.timeCounter = 0
    
    self.initialK = 3
    self.k = self.initialK
    
    self.W = np.eye(self.n)
    self.y = np.zeros((self.n,1))
    self.d = np.full((self.n,1),0.01)
    self.e  = np.zeros((1,self.n))
    
    self.ks = []
    self.ys = []
    
    self.L = []
    self.S = []
    
    self.reconstructionError = []
    
    self.energyX = []
    self.energyY = []
    self.ratios = []
    self.sumXSq = 0
    self.sumYSq = 0
    
#    self.ARs = []
#    self.windowSize = 20
#    self.predictions = []        
#    for j in range(self.n):
#        self.ARs.append(AR(self.windowSize))
        
    self.waitCounterReset = 150
    self.waitCounter = self.waitCounterReset
    
    self.dof = self.n
    self.prob = 0.95
    self.critical = stats.chi2.ppf(self.prob, self.dof)
    
  def handleRecord(self, inputData):
    """
    
    """
    self.waitCounter -= 1
    self.waitCounter = max(self.waitCounter,0)
    inputData = [inputData['v_{}'.format(i)] for i in range(len(inputData)-1)]
    #scaling step
    assert (len(self.inputMin) == len(inputData)), 'normalization error, len diff'
    
    inputData = (inputData-self.inputMean)/(self.inputStd)
    xn = inputData[:]
    
    t = self.timeCounter
    
    self.y = np.zeros((self.n,1))
    
    for i in range(self.k):
        #project xn onto corresponding basis vector
        self.y[i] = mm(xn,self.W[i:i+1,:].T)
        #calculate the energy of the data
        self.d[i] = self.lam*self.d[i]+self.y[i]**2
        #calculate the error
        e = xn - self.y[i]*self.W[i:i+1,:]
        #update w_i
        self.W[i:i+1,:] = self.W[i:i+1,:] + 1/self.d[i]*self.y[i]*e
        #continue with the remainder of xn
        xn = xn - self.y[i]*self.W[i:i+1,:]
        #normalize W - actually not needed when using QR() later on
        self.W[i:i+1,:] = self.W[i:i+1,:]/LA.norm(self.W[i:i+1,:])
    #orthogonalize used subspace vectors with Gram-Schmidt
    ortho, _ = np.linalg.qr(self.W[:self.k,:].T)
    self.W[:self.k,:] = ortho.T.copy()
    
    lHolder = np.full((1,self.n),0.0)
    #low dimensional representation
    lHolder[0,:self.k] = mm(inputData,self.W[:self.k,:].T)
    #reconstructed
    sHolder = (mm(lHolder[0,:self.k],self.W[:self.k,:]))
    
    if (t==0):
        self.L = lHolder
        self.S = sHolder
    else:
        self.L = np.vstack((self.L,lHolder))
        self.S = np.vstack((self.S,sHolder))

    self.sumXSq = self.lam * self.sumXSq + np.sum(inputData**2)
    self.sumYSq = self.lam * self.sumYSq +  np.sum(self.y**2)
    
    if (t==0):
        self.energyX.append(self.sumXSq/1.)
        self.energyY.append(self.sumYSq/1.)
    else:
        self.energyX.append(self.sumXSq)
        self.energyY.append(self.sumYSq)
        
    self.ratios.append(self.energyY[-1]/self.energyX[-1])
    self.reconstructionError.append(np.sum((inputData-self.S[t])**2))
    
    if self.energyY[-1] < self.fE*self.energyX[-1]:
        if self.k < self.n and self.waitCounter == 0:
            self.k += 1
            self.W[self.k:self.k+1,:] = np.eye(self.n)[self.k:self.k+1,:]
            self.waitCounter = self.waitCounterReset
    if self.energyY[-1] > self.FE*self.energyX[-1]:
        if self.k > 1 and self.waitCounter == 0:
            self.k -= 1
            self.waitCounter = self.waitCounterReset
    
    self.timeCounter += 1
       
    if (self.reconstructionError[-1] / self.critical > 0.33):
        return [self.reconstructionError[-1] / self.critical]
    else:
        return [0.0]

























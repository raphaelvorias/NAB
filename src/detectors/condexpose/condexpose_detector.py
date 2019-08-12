import numpy as np
from numpy import linalg as LA

from sklearn.kernel_approximation import RBFSampler

from detectors.base import AnomalyDetector

def mm(a,b):
    return np.matmul(a,b)
def inv(a):
    return np.linalg.inv(a)

class VRLS4(object): 
	'''Implemented using:
	Isukapalli, Yogananda. 
	“Recursive Least-Squares Adaptive Filters.” 
	February 21, 2019. 
	https://www.ece.ucsb.edu/~yoga/Adapt/P9_Recursive_Least_Squares.pdf.
	'''

    def __init__(self, n):
        self.P = 0.001 * np.eye(n)
        self.W = np.zeros((n,n))
        self.lam = 0.99
        self.invLam = 1/self.lam
        
    def update(self,U,D):
        pi = mm(self.P,U)
        K = pi/(self.lam+mm(U.T,pi))
        epsi = D - mm(self.W.T,U)
        self.W = self.W + mm(K,epsi.T)
        self.P = self.invLam*self.P - self.invLam*mm(K,mm(U.T,self.P))
        
    def getCovar(self):
        return self.W

class CondexposeDetector(AnomalyDetector):

  """ This is a modified EXPoSE detector that integrates a conditional
  temporal relation between two consequtive inputs.
  """

  def __init__(self, *args, **kwargs):
    super(CondexposeDetector, self).__init__(*args, **kwargs)

    self.kernel = None
    self.timestep = 0

  def initialize(self, gamma=None, fourierFeatures=None):
    """Initializes RBFSampler for the detector"""
    if gamma is None:
        self.gamma = 0.1
    else:
        self.gamma = gamma
    if fourierFeatures is None:
        self.fourierFeatures = 50
    else:
        self.fourierFeatures = fourierFeatures
        
    print('parameters -- gamma={} fourierFeatures={}'.format(self.gamma, self.fourierFeatures))

    self.kernel = RBFSampler(gamma=self.gamma, n_components=self.fourierFeatures, random_state=5)
    self.r = VRLS4(self.fourierFeatures)
    self.x_t = None

  def handleRecord(self, inputData):
    """ Returns a list [anomalyScore] calculated using a kernel based
    similarity method described in the comments below"""
    
    
    # Transform the input by approximating feature map of a Radial Basis
    # Function kernel using Random Kitchen Sinks approximation
    inputData = [inputData['v_{}'.format(i)] for i in range(len(inputData)-1)]
    inputData = (inputData-self.inputMin)/(self.inputMax-self.inputMin)
    #scaling step
    #todo: take outside and normalize all columns on their own
    assert (len(self.inputMin) == len(inputData)), 'normalization error, len diff'
    
    y_t = self.kernel.fit_transform(np.asarray([inputData]))
    
    if self.timestep == 0:
        self.x_t = y_t.copy()
    
    conditional_mean = (np.matmul(self.x_t,self.r.getCovar()))
    if i > 1:
        conditional_mean = conditional_mean/LA.norm(conditional_mean)
    anomalyScore = np.asscalar(1 - np.inner(y_t, conditional_mean))
    
    self.r.update(self.x_t.T,y_t.T)
    self.x_t = y_t.copy()
    
    self.timestep += 1

    return [anomalyScore]

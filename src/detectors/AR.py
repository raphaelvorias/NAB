# -*- coding: utf-8 -*-
'''adapted from
Yi, B.-K., and A Biliris. 
“Online Data Mining for Co-Evolving Time Sequences,” 1999, 26.
'''

import numpy as np

def mm(a,b):
    return np.matmul(a,b)
def inv(a):
    return np.linalg.inv(a)

class AR(object): 
    def __init__(self,windowSize):
        self.e = 0.01
        self.windowSize = windowSize
        self.history = np.zeros((windowSize,1))
        self.G = self.e*np.eye(windowSize)
        self.b = np.full((windowSize,1),0.1)
        
    def feed(self,x):
        J = inv(1+mm(self.history.T,mm(self.G,self.history)))
        K = mm(self.history,self.history.T)
        L = mm(self.G,mm(K,self.G))
        self.G = self.G - J[0,0]*L
        self.b = self.b - mm(self.G,self.history)*(mm(self.history.T,self.b)-x)
        self.history = np.roll(self.history,-1)
        self.history[-1] = x
        
    def predict(self):
        return mm(self.history.T,self.b)[0,0]
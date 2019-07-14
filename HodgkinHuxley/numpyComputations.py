import numpy as np

class numpyComputations(object):
    def __init__(self, Er=-54.387, ENa=50, EK=-77, Cm=1, gl=0.3, gNa=120, gK=36):
        self.Er = Er
        self.ENa = ENa
        self.EK = EK
        self.Cm = Cm
        self.gl = gl
        self.gNa = gNa
        self.gK = gK
        pass
    
    def alpha(self, gateType, Vm):
        if gateType == "n":
            a = 0.01
            b = 0.55
            c = 10
            d = 11/2
            e = 1
            return -(a*Vm + b)/(np.exp(-(Vm/c) - d)-e)
        elif gateType == "m":
            a = 0.1
            b = 40
            c = 1
            d = 40
            e = 10
            return (a * (Vm + b) / (c-np.exp(-(Vm + d)/e)))
        elif gateType == "h":
            a = 0.07
            b = 65
            c = 20
            return (a * np.exp(-(Vm+b)/c))
        
    def beta(self, gateType, Vm):
        if gateType == "n":
            a = 0.125
            b = 65
            c = 80
            return (a * np.exp(-(Vm + b)/c))
        elif gateType == "m":
            a = 4
            b = 65
            c = 18
            return (a * np.exp(-(Vm + b)/c))
        elif gateType == "h":
            a = 1
            b = 1
            c = 35
            d = 10
            return (a / (b + np.exp(-(Vm+c)/d)))
    
    def gate(self, alpha, beta):
        return (alpha/(alpha+beta))
    
    def gateRate(self, gateValue, alpha, beta):
        return (alpha*(1-gateValue)+beta*gateValue)

    def current(self, Vm, Iinj):
        mAlpha = self.alpha("m", Vm)
        mBeta = self.beta("m", Vm)
        m = self.gate(mAlpha, mBeta)
        hAlpha = self.alpha("h", Vm)
        hBeta = self.beta("h", Vm)
        h = self.gate(hAlpha, hBeta)
        nAlpha = self.alpha("n", Vm)
        nBeta = self.beta("n", Vm)
        n = self.gate(nAlpha, nBeta)
        return Vm, m, h, n
    
    def shaper(self, value, newShape):
        return np.reshape(np.asarray(value), newShape)
    
    def elementWiseMultiply(self, array1, array2):
        return np.multiply(array1, array2)
    
    def elementsSum(self, array):
        return np.sum(array)
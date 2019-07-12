import numpy as np
import tensorflow as tf
class model(object):
    def __init__(self, Er=-54.387, ENa=50, EK=-77, Cm=1, gl=0.3, gNa=120, gK=36):
        self.Er = Er
        self.ENa = ENa
        self.EK = EK
        self.Cm = Cm
        self.gl = gl
        self.gNa = gNa
        self.gK = gK
        pass
    
    # TENSOR FLOW FUNCTIONS:
    def computeTensorAlphas(self, gateType, membranePotential):
        if gateType == "n":
            a = tf.constant(0.01, dtype=tf.float32)
            b = tf.constant(0.55, dtype=tf.float32)
            c = tf.constant(10, dtype=tf.float32)
            d = tf.constant(11/2, dtype=tf.float32)
            e = tf.constant(1, dtype=tf.float32)
            return -(a*membranePotential + b)/(tf.exp(-(membranePotential/c) - d)-e)
        elif gateType == "m":
            a = tf.constant(0.1, dtype=tf.float32)
            b = tf.constant(40, dtype=tf.float32)
            c = tf.constant(1, dtype=tf.float32)
            d = tf.constant(40, dtype=tf.float32)
            e = tf.constant(10, dtype=tf.float32)
            return (a * (membranePotential + b) / (c-tf.exp(-(membranePotential + d)/e)))
        elif gateType == "h":
            a = tf.constant(0.07, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(20, dtype=tf.float32)
            return (a * tf.exp(-(membranePotential+b)/c))
    
    def computeTensorBetas(self, gateType, membranePotential):
        if gateType == "n":
            a = tf.constant(0.125, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(80, dtype=tf.float32)
            return (a * tf.exp(-(membranePotential + b)/c))
        elif gateType == "m":
            a = tf.constant(4, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(18, dtype=tf.float32)
            return (a * tf.exp(-(membranePotential + b)/c))
        elif gateType == "h":
            a = tf.constant(1, dtype=tf.float32)
            b = tf.constant(1, dtype=tf.float32)
            c = tf.constant(35, dtype=tf.float32)
            d = tf.constant(10, dtype=tf.float32)
            return (a / (b + tf.exp(-(membranePotential+c)/d)))
    
    def computeTensorGateValues(self, alpha, beta):
        return (alpha/(alpha+beta))
    
    def computeTensorGateRates(self, gateValue, alpha, beta):
        return (alpha*(tf.constant(1, dtype=tf.float32)-gateValue)+beta*gateValue)
    
    def futureTensorStates(self, membranePotential, injectedCurrent, Cm, gNa, gK, gl, ENa, EK, Er):
        nAlpha = self.computeTensorAlphas("n", membranePotential)
        mAlpha = self.computeTensorAlphas("m", membranePotential)
        hAlpha = self.computeTensorAlphas("h", membranePotential)
        nBeta = self.computeTensorBetas("n", membranePotential)
        mBeta = self.computeTensorBetas("m", membranePotential)
        hBeta = self.computeTensorBetas("h", membranePotential)
        n = self.computeTensorGateValues(nAlpha, nBeta)
        m = self.computeTensorGateValues(mAlpha, mBeta)
        h = self.computeTensorGateValues(hAlpha, hBeta)
        dvBydt = (1/Cm) * (injectedCurrent - gNa*tf.pow(m, tf.constant(3, dtype=tf.float32))*h*(membranePotential - ENa) - gK*tf.pow(n, tf.constant(4, dtype=tf.float32))*(membranePotential - EK) - gl*(membranePotential - Er))
        return dvBydt

    def equilibrium(self):
        dvBydt = tf.constant(0, dtype=tf.float32)
        Er = tf.constant(self.Er, dtype=tf.float32)
        ENa = tf.constant(self.ENa, dtype=tf.float32)
        EK = tf.constant(self.EK, dtype=tf.float32)
        Cm = tf.constant(self.Cm, dtype=tf.float32)
        gl = tf.constant(self.gl, dtype=tf.float32)
        gNa = tf.constant(self.gNa, dtype=tf.float32)
        gK = tf.constant(self.gK, dtype=tf.float32)
        Iinj = tf.constant(0, dtype=tf.float32)
        v = tf.Variable(0, dtype=tf.float32)
        def objectiveFunction():
            dvBydtComputed = self.futureTensorStates(membranePotential = v, injectedCurrent=0, Cm=Cm, gNa=gNa, gK=gK, gl=gl, ENa=ENa, EK=EK, Er=Er)
            return (dvBydt - dvBydtComputed)
        def costFunction():
            error = objectiveFunction()
            return tf.reduce_mean(tf.square(error)) + tf.reduce_sum(tf.square(error))
        cost = costFunction()
        optimize = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        def train(iterations, costTolerance=1E-8):
            for i in range(iterations):
                if i%1000==0:
                    trainCost, currentV = sess.run([cost, v])
                    nAlpha = self.computeNumpyAlphas("n", currentV)
                    mAlpha = self.computeNumpyAlphas("m", currentV)
                    hAlpha = self.computeNumpyAlphas("h", currentV)
                    nBeta = self.computeNumpyBetas("n", currentV)
                    mBeta = self.computeNumpyBetas("m", currentV)
                    hBeta = self.computeNumpyBetas("h", currentV)
                    n = self.computeNumpyGateValues(nAlpha, nBeta)
                    m = self.computeNumpyGateValues(mAlpha, mBeta)
                    h = self.computeNumpyGateValues(hAlpha, hBeta)
                    print("Iteration:", i, " cost:", trainCost, " @ v:", currentV, " m:", m, " h:", h, " n:", n)
                    if trainCost < costTolerance:
                        return currentV, m, n, h
                sess.run(optimize)
            pass
        v_eq, m_eq, h_eq, n_eq = train(100000)
        print("Equilibrum @ v:", v_eq, " m:", m_eq, " h:", h_eq, " n:", n_eq)
        self.statesTensorJacobian(v_eq, m_eq, h_eq, n_eq)
        pass

    # NUMPY FUNCTIONS:
    def computeNumpyJacobian(self):
        pass
    
    def computeNumpyAlphas(self, gateType, membranePotential):
        if gateType == "n":
            a = 0.01
            b = 0.55
            c = 10
            d = 11/2
            e = 1
            return -(a*membranePotential + b)/(np.exp(-(membranePotential/c) - d)-e)
        elif gateType == "m":
            a = 0.1
            b = 40
            c = 1
            d = 40
            e = 10
            return (a * (membranePotential + b) / (c-np.exp(-(membranePotential + d)/e)))
        elif gateType == "h":
            a = 0.07
            b = 65
            c = 20
            return (a * np.exp(-(membranePotential+b)/c))
    
    def computeNumpyBetas(self, gateType, membranePotential):
        if gateType == "n":
            a = 0.125
            b = 65
            c = 80
            return (a * np.exp(-(membranePotential + b)/c))
        elif gateType == "m":
            a = 4
            b = 65
            c = 18
            return (a * np.exp(-(membranePotential + b)/c))
        elif gateType == "h":
            a = 1
            b = 1
            c = 35
            d = 10
            return (a / (b + np.exp(-(membranePotential+c)/d)))
    
    def computeNumpyGateValues(self, alpha, beta):
        return (alpha/(alpha+beta))
    
    def computeNumpyGateRates(self, gateValue, alpha, beta):
        return (alpha*(1-gateValue)+beta*gateValue)
    
    def futureNumpyStates(self, membranePotential, injectedCurrent, Cm, gNa, gK, gl, ENa, EK, Er):
        nAlpha = self.computeNumpyAlphas("n", membranePotential)
        mAlpha = self.computeNumpyAlphas("m", membranePotential)
        hAlpha = self.computeNumpyAlphas("h", membranePotential)
        nBeta = self.computeNumpyBetas("n", membranePotential)
        mBeta = self.computeNumpyBetas("m", membranePotential)
        hBeta = self.computeNumpyBetas("h", membranePotential)
        n = self.computeNumpyGateValues(nAlpha, nBeta)
        m = self.computeNumpyGateValues(mAlpha, mBeta)
        h = self.computeNumpyGateValues(hAlpha, hBeta)
        dvBydt = (1/Cm) * (injectedCurrent - gNa*np.pow(m, 3)*h*(membranePotential - ENa) - gK*np.pow(n, 4)*(membranePotential - EK) - gl*(membranePotential - Er))
        return dvBydt

    
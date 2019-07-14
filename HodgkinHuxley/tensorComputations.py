import tensorflow as tf
from HodgkinHuxley.numpyComputations import numpyComputations

class tensorComputations(object):
    def __init__(self, Er=-54.387, ENa=50, EK=-77, Cm=1, gl=0.3, gNa=120, gK=36):
        self.Er = tf.constant(Er, dtype=tf.float32)
        self.ENa = tf.constant(ENa, dtype=tf.float32)
        self.EK = tf.constant(EK, dtype=tf.float32)
        self.Cm = tf.constant(Cm, dtype=tf.float32)
        self.gl = tf.constant(gl, dtype=tf.float32)
        self.gNa = tf.constant(gNa, dtype=tf.float32)
        self.gK = tf.constant(gK, dtype=tf.float32)
        self.npComps = numpyComputations(Er=-54.387, ENa=50, EK=-77, Cm=1, gl=0.3, gNa=120, gK=36)
        pass
    
    def alpha(self, gateType, Vm):
        if gateType == "n":
            a = tf.constant(0.01, dtype=tf.float32)
            b = tf.constant(0.55, dtype=tf.float32)
            c = tf.constant(10, dtype=tf.float32)
            d = tf.constant(11/2, dtype=tf.float32)
            e = tf.constant(1, dtype=tf.float32)
            return -(a*Vm + b)/(tf.exp(-(Vm/c) - d)-e)
        elif gateType == "m":
            a = tf.constant(0.1, dtype=tf.float32)
            b = tf.constant(40, dtype=tf.float32)
            c = tf.constant(1, dtype=tf.float32)
            d = tf.constant(40, dtype=tf.float32)
            e = tf.constant(10, dtype=tf.float32)
            return (a * (Vm + b) / (c-tf.exp(-(Vm + d)/e)))
        elif gateType == "h":
            a = tf.constant(0.07, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(20, dtype=tf.float32)
            return (a * tf.exp(-(Vm+b)/c))
    
    def beta(self, gateType, Vm):
        if gateType == "n":
            a = tf.constant(0.125, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(80, dtype=tf.float32)
            return (a * tf.exp(-(Vm + b)/c))
        elif gateType == "m":
            a = tf.constant(4, dtype=tf.float32)
            b = tf.constant(65, dtype=tf.float32)
            c = tf.constant(18, dtype=tf.float32)
            return (a * tf.exp(-(Vm + b)/c))
        elif gateType == "h":
            a = tf.constant(1, dtype=tf.float32)
            b = tf.constant(1, dtype=tf.float32)
            c = tf.constant(35, dtype=tf.float32)
            d = tf.constant(10, dtype=tf.float32)
            return (a / (b + tf.exp(-(Vm+c)/d)))
        
    def gate(self, alpha, beta):
        return alpha/(alpha + beta)
    
    def gateRate(self, gate, alpha, beta):
        return alpha*(tf.constant(1, dtype=tf.float32)-gate)+beta*gate
    
    def future(self, Vm, Iinj, m=None, h=None, n=None):
        mAlpha = self.alpha("m", Vm)
        mBeta = self.beta("m", Vm)
        if m == None:
            m = self.gate(mAlpha, mBeta)
        else:
            m = m
        hAlpha = self.alpha("h", Vm)
        hBeta = self.beta("h", Vm)
        if h == None:
            h = self.gate(hAlpha, hBeta)
        else:
            h = h
        nAlpha = self.alpha("n", Vm)
        nBeta = self.beta("n", Vm)
        if n == None:
            n = self.gate(nAlpha, nBeta)
        else:
            n = n
        VmFuture = (1/self.Cm)*(Iinj - self.gNa*tf.pow(m, 3)*h*(Vm - self.ENa) - self.gK*tf.pow(n, 4)*(Vm - self.EK) - self.gl*(Vm - self.Er))
        mFuture = self.gateRate(m, mAlpha, mBeta)
        hFuture = self.gateRate(h, hAlpha, hBeta)
        nFuture = self.gateRate(n, nAlpha, nBeta)
        return VmFuture, mFuture, hFuture, nFuture

    def equilibrium(self):
        VmFuture = tf.constant(0, dtype=tf.float32)
        Iinj = tf.constant(0, dtype=tf.float32)
        Vm = tf.Variable(0, dtype=tf.float32)
        VmFutureComputed, _, _, _ = self.future(Vm, Iinj)
        cost = tf.reduce_mean(tf.square(tf.subtract(VmFuture, VmFutureComputed)))
        iterations = 100000
        costTolerance = 1E-8
        VmCurrent = 0
        optimize = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                if i % 1000 == 0:
                    trainCost, VmCurrent = sess.run([cost, Vm])
                    print("Iteration:", i, " cost:", trainCost, " @ Vm:", VmCurrent)
                    if trainCost < costTolerance:
                        break
                sess.run(optimize)
        _, mCurrent, hCurrent, nCurrent = self.npComps.current(VmCurrent, 0)
        print("The equilibria are at: ", VmCurrent, mCurrent, hCurrent, nCurrent)
        return VmCurrent, mCurrent, hCurrent, nCurrent

    def jacobian(self, Iinj=0, VmEquilibrium=0, mEquilibrium=0, hEquilibrium=0, nEquilibrium=0):
        Vm = tf.Variable(VmEquilibrium, dtype=tf.float32)
        m = tf.Variable(mEquilibrium, dtype=tf.float32)
        h = tf.Variable(hEquilibrium, dtype=tf.float32)
        n = tf.Variable(nEquilibrium, dtype=tf.float32)
        Iinj = tf.constant(Iinj, dtype=tf.float32)
        VmFuture, mFuture, hFuture, nFuture = self.future(Vm, Iinj, m, h, n)
        VmGradients = tf.gradients(ys=VmFuture, xs=[Vm, m, h, n])
        mGradients = tf.gradients(ys=mFuture, xs=[Vm, m])
        hGradients = tf.gradients(ys=hFuture, xs=[Vm, h])
        nGradients = tf.gradients(ys=nFuture, xs=[Vm, n])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _, _, _, _, VmGrads, mGrads, hGrads, nGrads= sess.run([VmFuture, mFuture, hFuture, nFuture, VmGradients, mGradients, hGradients, nGradients])
        return [VmGrads, [mGrads[0], mGrads[1], 0, 0], [hGrads[0], 0, hGrads[1], 0], [nGrads[0], 0, 0, nGrads[0]]]

    def lyapunovCandidate(self, jacobian):
        A = tf.constant(jacobian, dtype=tf.float32)
        Q = tf.cast(tf.diag([1,1,1,1]),dtype=tf.float32)
        X = tf.Variable(tf.zeros([4, 4]), dtype=tf.float32)
        cost = tf.reduce_mean(tf.square(self.lyapunovEquation(A, X, Q)))
        optimize = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
        iterations = 100000
        costTolerance = 1E-8
        candidate = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                if i % 1000 == 0:
                    trainCost, candidate = sess.run([cost, X])
                    print("Iteration:", i, " cost:", trainCost, " @ candidate:")
                    print(candidate)
                    if trainCost < costTolerance:
                        return candidate
                sess.run(optimize)
        return candidate

    def lyapunovEquation(self, A, X, Q):
        A_H = tf.transpose(tf.conj(A))
        return tf.add(tf.subtract(tf.multiply(tf.multiply(A, X), A_H), X), Q)

    def lyapunovFunction(self, candidate, VmStart=25, IinjStart=0):
        VmCurrent, mCurrent, hCurrent, nCurrent = self.npComps.current(VmStart, IinjStart)
        states = tf.constant([VmCurrent, mCurrent, hCurrent, nCurrent], shape=[4, 1], dtype=tf.float32)
        Iinj = tf.constant(IinjStart, dtype=tf.float32)
        surface = tf.linalg.matmul(tf.transpose(states), tf.linalg.matmul(candidate, states))
        gradients = tf.gradients(ys=surface, xs=states)
        futures = self.future(states[0, 0], Iinj, states[1, 0], states[2, 0], states[3, 0])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            currents, surf, grads, futures = sess.run([states, surface, gradients, futures])
        grads = self.npComps.shaper(grads[0], [1, 4])
        futures = self.npComps.shaper(futures, [1, 4])
        surfaceGradient = self.npComps.elementsSum(self.npComps.elementWiseMultiply(grads, futures))
        print(surf)
        if surf[0] > 0 and surfaceGradient < 0:
            print("The system is stable at Vm:", VmStart)
        else:
            print("The system is unstable at Vm", VmStart)
        pass
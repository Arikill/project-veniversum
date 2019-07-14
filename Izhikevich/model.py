import tensorflow as tf
import numpy as np

class model(object):
    def __init__(self, Cm = 100, Er = -60, Et = -40, Ep = 35, k = 0.7, a = 0.03, b = -2, c = -50, d = 100):
        self.Cm = tf.constant(Cm, tf.float32)
        self.Er = tf.constant(Er, tf.float32)
        self.Et = tf.constant(Et, tf.float32)
        self.Ep = tf.constant(Ep, tf.float32)
        self.k = tf.constant(k, tf.float32)
        self.a = tf.constant(a, tf.float32)
        self.b = tf.constant(b, tf.float32)
        self.c = tf.constant(c, tf.float32)
        self.d = tf.constant(d, tf.float32)
        pass
    
    def future(self, Vm, u, Iinj):
        VmNext = (self.k/self.Cm)*(Vm - self.Er)*(Vm - self.Et) - (1/self.Cm)*u + (1/self.Cm)*Iinj
        uNext = self.a*(self.b*(Vm - self.Er) - u)
        VmFuture = tf.cond(VmNext >= self.Ep, lambda: self.c, lambda: VmNext)
        uFuture = tf.cond(VmNext >= self.Ep, lambda: uNext+self.d, lambda: uNext)
        return VmFuture, uFuture
    
    def equilibrium(self):
        VmFuture = tf.constant(0, dtype=tf.float32)
        uFuture = tf.constant(0, dtype=tf.float32)
        Iinj = tf.constant(0, dtype=tf.float32)
        Vm = tf.Variable(0, dtype=tf.float32)
        u = tf.Variable(0, dtype=tf.float32)
        VmFutureComputed, uFutureComputed = self.future(Vm=Vm, u=u, Iinj=Iinj)
        cost = tf.reduce_mean([tf.square(tf.subtract(VmFuture, VmFutureComputed)), tf.square(tf.subtract(uFuture, uFutureComputed))])
        iterations = 100000
        costTolerance = 1E-20
        optimize = tf.train.AdamOptimizer(learning_rate=5).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                if i % 1000 == 0:
                    trainCost, VmCurrent, uCurrent = sess.run([cost, Vm, u])
                    print("Iteration:", i, " cost:", trainCost, " @ Vm:", VmCurrent, " u:", uCurrent)
                    if trainCost < costTolerance:
                        break
                sess.run(optimize)
        print("The equilibria are @ Vm:", VmCurrent, " u:", uCurrent)
        return VmCurrent, uCurrent
    
    def jacobian(self, Iinj = 0, VmEquilibrium = 0, uEquilibrium=0):
        Vm = tf.Variable(VmEquilibrium, dtype=tf.float32)
        u = tf.Variable(uEquilibrium, dtype=tf.float32)
        Iinj = tf.constant(Iinj, dtype=tf.float32)
        VmFuture, uFuture = self.future(Vm, u, Iinj)
        VmGradients = tf.gradients(ys=VmFuture, xs=[Vm, u])
        uGradients = tf.gradients(ys=uFuture, xs=[Vm, u])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _, _, VmGrads, uGrads = sess.run([VmFuture, uFuture, VmGradients, uGradients])
        gradients = [VmGrads, uGrads]
        jacobianEigenValues, _ = np.linalg.eig(gradients)
        print(jacobianEigenValues)
        if (jacobianEigenValues.real < 0).all():
            print("The space time point Vm:", VmEquilibrium, " and u:", uEquilibrium, " is asymptotically stable.")
        else:
            print("The space time point Vm:", VmEquilibrium, " and u:", uEquilibrium, " is a unstable.")
        return gradients

    def lyapunovEquation(self, A, X, Q):
        A_H = tf.transpose(tf.conj(A))
        return tf.add(tf.subtract(tf.multiply(tf.multiply(A, X), A_H), X), Q)

    def lyapunovCandidate(self, jacobian):
        A = tf.constant(jacobian, dtype=tf.float32)
        Q = tf.cast(tf.diag([1, 1]),dtype=tf.float32)
        X = tf.Variable(tf.zeros([2, 2]), dtype=tf.float32)
        cost = tf.reduce_mean(tf.square(self.lyapunovEquation(A, X, Q)))
        optimize = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
        iterations = 100000
        costTolerance = 1E-20
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
    
    def lyapunovFunction(self, candidate, VmStart=-40, uStart=0, IinjStart=10):
        Vm = tf.Variable(VmStart, dtype=tf.float32)
        u = tf.Variable(uStart, dtype=tf.float32)
        Iinj = tf.constant(IinjStart, dtype=tf.float32)
        surface = tf.linalg.matmul(tf.transpose([[Vm], [u]]), tf.linalg.matmul(candidate, [[Vm], [u]]))
        gradients = tf.gradients(ys=surface, xs=[Vm, u])
        futures = self.future(Vm, u, Iinj)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            VmCurrent, uCurrent, surf, grads, futures = sess.run([Vm, u, surface, gradients, futures])
        grads = np.reshape(grads, [1, 2])
        futures = np.reshape(futures, [1, 2])
        surfaceGradient = np.sum(np.multiply(grads, futures))
        print(surf)
        print(surfaceGradient)
        if surf[0] > 0 and surfaceGradient < 0:
            print("The system is stable at Vm:", VmCurrent)
        else:
            print("The system is unstable at Vm", VmCurrent)
        pass
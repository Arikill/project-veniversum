from HodgkinHuxley.tensorComputations import tensorComputations

tenComp = tensorComputations()
VmEq, mEq, hEq, nEq = tenComp.equilibrium()
J = tenComp.jacobian(Iinj=0, VmEquilibrium=VmEq, mEquilibrium=mEq, hEquilibrium=hEq, nEquilibrium=nEq)
C = tenComp.lyapunovCandidate(J)
print(tenComp.lyapunovFunction(C))
print(J)
from HodgkinHuxley.tensorComputations import tensorComputations

# tenComp = tensorComputations()
# VmEq, mEq, hEq, nEq = tenComp.equilibrium()
# J = tenComp.jacobian(Iinj=0, VmEquilibrium=VmEq, mEquilibrium=mEq, hEquilibrium=hEq, nEquilibrium=nEq)
# C = tenComp.lyapunovCandidate(J)
# print(tenComp.lyapunovFunction(C))
# print(J)
from Izhikevich.model import model
m = model()
VmEq, uEq = m.equilibrium()
J = m.jacobian(Iinj = 0, VmEquilibrium = VmEq, uEquilibrium=uEq)
C = m.lyapunovCandidate(J)
print(m.lyapunovFunction(C))
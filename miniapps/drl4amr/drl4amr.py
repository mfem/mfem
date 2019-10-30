from ctypes import *
from random import *
MFEM = cdll.LoadLibrary('./libdrl4amr.so')

class Ctrl(object):
    def __init__(self, order): self.obj = MFEM.Ctrl(order)
    def Compute(self): MFEM.Compute(self.obj)
    def Refine(self, el_to_refine): MFEM.Refine(self.obj, el_to_refine)
    def GetNE(self): return MFEM.GetNE(self.obj)
    def GetNorm(self): return MFEM.GetNorm(self.obj)
    MFEM.GetNorm.restype = c_double
    def GetNDofs(self): return MFEM.GetNDofs(self.obj)
    def GetImage(self): return MFEM.GetImage(self.obj)

order = 3
sim = Ctrl(order)

while sim.GetNorm() > 0.01:
    NE = sim.GetNE()
    sim.Compute()
    #sim.Refine(-1); # Will refine using the internal refiner
    sim.Refine(int(NE*random()))
    #sim.GetImage()

print "Done"

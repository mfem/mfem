from ctypes import *
from random import *
MFEM = cdll.LoadLibrary('./libdrl4amr.so')

class Ctrl(object):
    def __init__(self, order): self.obj = MFEM.Ctrl(order)
    MFEM.Ctrl.restype = c_void_p
    MFEM.Ctrl.argtypes = [c_int]
    def Compute(self): return MFEM.Compute(self.obj)
    MFEM.Compute.argtypes = [c_void_p]
    def Refine(self, el_to_refine): return MFEM.Refine(self.obj, el_to_refine)
    MFEM.Refine.argtypes = [c_void_p, c_int]
    def GetNE(self): return MFEM.GetNE(self.obj)       
    MFEM.GetNE.argtypes = [c_void_p] 
    def GetNorm(self): return MFEM.GetNorm(self.obj)
    MFEM.GetNorm.argtypes = [c_void_p]
    MFEM.GetNorm.restype = c_double
    def GetTrueVSize(self): return MFEM.GetTrueVSize(self.obj)
    MFEM.GetTrueVSize.argtypes = [c_void_p]
    def GetImage(self): MFEM.GetImage(self.obj)
    MFEM.GetImage.argtypes = [c_void_p]

order = 3
sim = Ctrl(order)

while sim.GetNorm() > 0.01:
    NE = sim.GetNE()
    sim.Compute()
    #sim.Refine(-1); # Will refine using the internal refiner
    sim.Refine(int(NE*random()))
    sim.GetImage()
    print("Norm: "+str(sim.GetNorm()))

print("Done, final norm: "+str(sim.GetNorm()))

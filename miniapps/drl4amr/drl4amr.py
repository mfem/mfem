from ctypes import *
MFEM = cdll.LoadLibrary('./libdrl4amr.so')

class Ctrl(object):
    def __init__(self, order): self.obj = MFEM.Ctrl(order)
    def Compute(self): MFEM.Compute(self.obj)
    def Refine(self): MFEM.Refine(self.obj)
    def Update(self): MFEM.Update(self.obj)
    def GetNDofs(self): return MFEM.GetNDofs(self.obj)
    def GetNorm(self): return MFEM.GetNorm(self.obj)
    MFEM.GetNorm.restype = c_double

order = 3
sim = Ctrl(order)

while sim.GetNorm() > 0.01:
    sim.Compute()
    sim.Refine()
    print(sim.Update())


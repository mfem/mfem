from ctypes import cdll
MFEM = cdll.LoadLibrary('./libdrl4amr.so')

class Ctrl(object):
    def __init__(self): self.obj = MFEM.Ctrl()
    def Compute(self): MFEM.Compute(self.obj)
    def Refine(self): MFEM.Refine(self.obj)
    def Update(self): MFEM.Update(self.obj)

sim = Ctrl()

sim.Compute()
sim.Refine()
sim.Update()

sim.Refine()
sim.Update()
sim.Compute()

sim.Refine()
sim.Update()
sim.Compute()

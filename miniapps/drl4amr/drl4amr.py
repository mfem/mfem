import numpy as np
from ctypes import *
from random import *
from PIL import Image

c_double_p = POINTER(c_double)

MFEM = cdll.LoadLibrary('./libdrl4amr.so')

def sign(f, args_t, res_t):
    f.restype = res_t
    f.argtypes = args_t

class Ctrl(object):
    def __init__(self, order): self.obj = MFEM.Ctrl(order)
    sign(MFEM.Ctrl, [c_int], c_void_p)

    def Compute(self): return MFEM.Compute(self.obj)
    sign(MFEM.Compute, [c_void_p], c_int)

    def Refine(self, el_to_refine): return MFEM.Refine(self.obj, el_to_refine)
    sign(MFEM.Refine, [c_void_p, c_int], c_int)

    def GetNE(self): return MFEM.GetNE(self.obj)
    sign(MFEM.GetNE, [c_void_p], c_int)

    def GetNorm(self): return MFEM.GetNorm(self.obj)
    sign(MFEM.GetNorm, [c_void_p], c_double)

    def GetNDofs(self): return MFEM.GetNDofs(self.obj)
    sign(MFEM.GetNDofs, [c_void_p], c_int)

    def GetImage(self): return MFEM.GetImage(self.obj)
    sign(MFEM.GetImage, [c_void_p], c_double_p)

    def GetImageSize(self): return MFEM.GetImageSize(self.obj)
    sign(MFEM.GetImageSize, [c_void_p], c_int)

    def GetImageX(self): return MFEM.GetImageX(self.obj)
    sign(MFEM.GetImageX, [c_void_p], c_int)

    def GetImageY(self): return MFEM.GetImageY(self.obj)
    sign(MFEM.GetImageY, [c_void_p], c_int)


# Orders 1 and 2 are supported with the current GetImage
order = 2

sim = Ctrl(order)

NE = sim.GetNE()
NX = sim.GetImageX()
NY = sim.GetImageY()
#print(NX,NY)

while sim.GetNorm() > 0.01:
    sim.Compute()

    #sim.Refine(-1); # Will refine using the internal refiner
    sim.Refine(int(NE*random()))

    image_d = sim.GetImage()
    #image_s = sim.GetImageSize()
    #print("image_s: " + str(image_s))

    # Get the data back, two ways:
    #data = np.fromiter(image_d, dtype=np.double, count=NX*NY) # copy
    # or:
    data = np.frombuffer((c_double*NX*NY).from_address(addressof(image_d.contents))) # address

    # Scale and convert the image
    image_f = (data * 255 / np.max(data)).astype('uint8')
    image = Image.fromarray(image_f.reshape(NX,NY),'L')
    image.save('image.jpg')
    print("Norm: "+str(sim.GetNorm()))

print("Done, final norm: "+str(sim.GetNorm()))

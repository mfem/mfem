import sys
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
    def __init__(self, order, seed): self.obj = MFEM.Ctrl(order, seed)
    sign(MFEM.Ctrl, [c_int, c_int], c_void_p)

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

    # Get the image of the solution of size GetNodeImageSize
    def GetImage(self): return MFEM.GetImage(self.obj)
    sign(MFEM.GetImage, [c_void_p], c_double_p)

    # Get the image of the elem id, of size GetElemImageSize
    def GetElemIdField(self): return MFEM.GetElemIdField(self.obj)
    sign(MFEM.GetElemIdField, [c_void_p], c_double_p)

    # Get the image of the elem depth
    def GetElemDepthField(self): return MFEM.GetElemDepthField(self.obj)
    sign(MFEM.GetElemDepthField, [c_void_p], c_double_p)

    # Get Node images sizes: GetNodeImageSize = GetNodeImageX * GetNodeImageY
    def GetNodeImageSize(self): return MFEM.GetNodeImageSize(self.obj)
    sign(MFEM.GetNodeImageSize, [c_void_p], c_int)

    def GetNodeImageX(self): return MFEM.GetNodeImageX(self.obj)
    sign(MFEM.GetNodeImageX, [c_void_p], c_int)

    def GetNodeImageY(self): return MFEM.GetNodeImageY(self.obj)
    sign(MFEM.GetNodeImageY, [c_void_p], c_int)

    # Get Elem images sizes: GetElemImageSize = GetElemImageX * GetElemImageY
    def GetElemImageSize(self): return MFEM.GetElemImageSize(self.obj)
    sign(MFEM.GetElemImageSize, [c_void_p], c_int)

    def GetElemImageX(self): return MFEM.GetElemImageX(self.obj)
    sign(MFEM.GetElemImageX, [c_void_p], c_int)

    def GetElemImageY(self): return MFEM.GetElemImageY(self.obj)
    sign(MFEM.GetElemImageY, [c_void_p], c_int)


seed = 0
order = 2

sim = Ctrl(order, seed)

NE = sim.GetNE()

NX = sim.GetNodeImageX()
NY = sim.GetNodeImageY()
print(NX,NY)

EX = sim.GetElemImageX()
EY = sim.GetElemImageY()
print(EX,EY)

while sim.GetNorm() > 0.01:
    sim.Compute()

    #sim.Refine(-1); # Will refine using the internal refiner
    sim.Refine(int(NE*random()))

    image_d = sim.GetImage()
#    field = sim.GetField()
    depth_d = sim.GetElemDepthField()
    id = sim.GetElemIdField()
    #image_s = sim.GetImageSize()
    #print("image_s: " + str(image_s))

    # Get the data back
    data = np.frombuffer((c_double*NX*NY).from_address(addressof(image_d.contents)))
    depth = np.frombuffer((c_double*EX*EY).from_address(addressof(depth_d.contents)))
    # np.set_printoptions(threshold = sys.maxsize)
    # print level

    # Scale and convert the image
    image_f = (data * 255 / np.max(data)).astype('uint8')
    image = Image.fromarray(image_f.reshape(NX,NY),'L')
    image.save('image.jpg')

    depth_f = (depth * 255 / np.max(data)).astype('uint8')
    depth_image = Image.fromarray(depth_f.reshape(EX,EY),'L')
    depth_image.save('depth.jpg')

    print("Norm: "+str(sim.GetNorm()))

print("Done, final norm: "+str(sim.GetNorm()))

import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_double
from ctypes import c_void_p
from ctypes import POINTER
from ctypes import addressof
from random import random
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

    # Get the image of the solution, of size GetImageSize
    def GetImage(self): return MFEM.GetImage(self.obj)
    sign(MFEM.GetImage, [c_void_p], c_double_p)

    # Get the image of the elem id, of size GetImageSize
    def GetIdField(self): return MFEM.GetIdField(self.obj)
    sign(MFEM.GetIdField, [c_void_p], c_double_p)

    # Get the image of the elem depth, of size GetImageSize
    def GetDepthField(self): return MFEM.GetDepthField(self.obj)
    sign(MFEM.GetDepthField, [c_void_p], c_double_p)

    # Get images sizes: GetImageSize = GetImageX * GetImageY
    def GetImageSize(self): return MFEM.GetImageSize(self.obj)
    sign(MFEM.GetImageSize, [c_void_p], c_int)

    def GetImageX(self): return MFEM.GetImageX(self.obj)
    sign(MFEM.GetImageX, [c_void_p], c_int)

    def GetImageY(self): return MFEM.GetImageY(self.obj)
    sign(MFEM.GetImageY, [c_void_p], c_int)


seed = 0
order = 2

sim = Ctrl(order, seed)

NE = sim.GetNE()
NX = sim.GetImageX()
NY = sim.GetImageY()
print(NX, NY)


while sim.GetNorm() > 0.01:
    sim.Compute()

    # sim.Refine(-1); # Will refine using the internal refiner
    sim.Refine(int(NE*random()))

    image_p = sim.GetImage()
    id_p = sim.GetIdField()
    depth_p = sim.GetDepthField()

    # Get the data back
    size_xy = c_double*NX*NY
    image_d = np.frombuffer(size_xy.from_address(addressof(image_p.contents)))
    id_d = np.frombuffer(size_xy.from_address(addressof(id_p.contents)))
    depth_d = np.frombuffer(size_xy.from_address(addressof(depth_p.contents)))

    # Scale and convert the image
    image_f = (image_d * 255 / np.max(image_d)).astype('uint8')
    image = Image.fromarray(image_f.reshape(NX, NY), 'L')
    image.save('image.jpg')

    id_f = (id_d * 255 / np.max(id_d)).astype('uint8')
    id = Image.fromarray(id_f.reshape(NX, NY), 'L')
    id.save('id.jpg')

    depth_f = (depth_d * 255 / np.max(depth_d)).astype('uint8')
    depth = Image.fromarray(depth_f.reshape(NX, NY), 'L')
    depth.save('depth.jpg')

    print("Norm: "+str(sim.GetNorm()))

print("Done, final norm: "+str(sim.GetNorm()))

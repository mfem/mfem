from ctypes import cdll
mfem = cdll.LoadLibrary('./libdrl4amr.so')

class Mfem(object):
    def __init__(self):
        self.obj = mfem.Foo_new()

    def bar(self):
        mfem.Foo_bar(self.obj)

f = Mfem()
f.bar() #and you will see "Hello" on the screen

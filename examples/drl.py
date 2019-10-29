import numpy
import random
import drl4amr

if __name__ == '__main__':
    drand = [random.random() for _ in range(0, 100)]
    np_drand = numpy.array(drand)
    print(np_drand)
    print("dot:")
    drl4amr.dot(np_drand)
    print("ex6:")
    drl4amr.ex6(np_drand)

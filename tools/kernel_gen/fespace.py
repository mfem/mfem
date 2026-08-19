import expr
from enum import Enum


class GeometryType(Enum):
    SEGMENT = 1
    TRIANGLE = 2
    SQUARE = 3
    TETRAHEDRON = 4
    CUBE = 5
    PRISM = 6
    PYRAMID = 7

# TODO: runtime specified vdims. Perhaps use 0 or -1 to specify runtime dim?

class FESpace(expr.Expr):
    def __init__(self, order: int, geom: GeometryType, vdims: int):
        self.order = order
        self.geom = geom
        super().__init__(args=[], width=1, height=vdims)


class H1Space(FESpace):
    def __init__(self, order: int, geom: GeometryType, vdims: int):
        super().__init__(order=order, geom=geom, vdims=vdims)


class L2Space(FESpace):
    def __init__(self, order: int, geom: GeometryType, vdims: int, integral: bool):
        super().__init__(order=order, geom=geom, vdims=vdims)
        self.integral = integral


class RTSpace(FESpace):
    def __init__(self, order: int, geom: GeometryType):
        # TODO: space dims?
        if geom == GeometryType.TRIANGLE or geom == GeometryType.SQUARE:
            vdim = 2
        elif geom == GeometryType.SEGMENT:
            raise RuntimeError("TODO: R1D")
        else:
            vdim = 3
        super().__init__(order=order, geom=geom, vdims=vdim)


class NDSpace(FESpace):
    def __init__(self, order: int, geom: GeometryType):
        # TODO: space dims?
        if geom == GeometryType.TRIANGLE or geom == GeometryType.SQUARE:
            vdim = 2
        elif geom == GeometryType.SEGMENT:
            raise RuntimeError("TODO: ND_SEGMENT")
        else:
            vdim = 3
        super().__init__(order=order, geom=geom, vdims=vdim)

        

class Jacobian(expr.Expr):
    """
    Element spatial Jacobian
    """

    def __init__(self, ndims: int, sdims: int):
        super().__init__(args=[], width=ndims, height=sdims)

    def __str__(self):
        return "J"

    def Metric(self):
        """
        if rectangular, Sqrt((self.T() @ self).Det())
        else self.Det()
        """
        if self.width == self.height:
            return self.Det()
        return expr.Sqrt((self.T() @ self).Det())



class BilinearForm(expr.Expr):
    """
    Represents mixed and non-mixed bilinear form integrators
    """
    def __init__(self, a: expr.Expr, b: expr.Expr):
        """
        Bilinear form (a, b). @a a must be a row vector, and @a b must be a column vector.
        """
        if a.width != b.height or a.height != 1 or b.width != 1:
            raise RuntimeError(
                f"invalid bilinear form argument shapes: ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        super().__init__(args=[a, b], width=1, height=1)

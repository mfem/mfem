class Expr:
    """
    (Matrix) Expression base class.
    Scalars have shape (1,1).
    Vectors can have shape (height,1) or (1,width).
    """

    def __init__(self, args: list, width: int, height: int):
        self.args = args
        self.width = width
        self.height = height
        self.shape = (height, width)

    def __add__(self, other):
        return AddExpr(self, other)

    def __sub__(self, other):
        return SubExpr(self, other)

    def __mul__(self, other):
        return MulExpr(self, other)

    def __matmul__(self, other):
        return MatMulExpr(a, other)

    def __neg__(self):
        return NegExpr(self)

    def __pos__(self):
        return self

    def __pow__(self, other):
        return PowExpr(self, other)

    def __truediv__(self, other):
        return DivExpr(self, other)

    def __getitem__(self, idcs):
        return GetExpr(self, idcs)

    def T(self):
        """
        Transpose
        """
        return TransposeExpr(self)

    def Adj(self):
        """
        Matrix adjoint
        """
        return AdjointExpr(self)

    def Inv(self):
        """
        Matrix inverse
        """
        return InverseExpr(self)

    def Det(self):
        """
        Matrix determinant
        """
        return DetExpr(self)


class TransposeExpr(Expr):
    def __init__(self, v: Expr):
        super().__init__(args=[v], width=v.height, height=v.width)


class AdjointExpr(Expr):
    def __init__(self, v: Expr):
        super().__init__(args=[v], width=v.height, height=v.width)


class InvExpr(Expr):
    def __init__(self, v: Expr):
        if v.width != v.height:
            raise RuntimeError("Can only invert square matrices")
        super().__init__(args=[v], width=v.height, height=v.width)


class DetExpr(Expr):
    def __init__(self, v: Expr):
        if v.width != v.height:
            raise RuntimeError("Can only take determinant of square matrices")
        super().__init__(args=[v], width=1, height=1)


class AddExpr(Expr):
    def __init__(self, a: Expr, b: Expr):
        if a.width * a.height == 1:
            # scalar broadcast into all components of b
            width = b.width
            height = b.height
        elif a.width != b.width or a.height != b.height:
            raise RuntimeError(
                f"Invalid shapes ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        else:
            width = a.width
            height = a.height
        super().__init__(args=[a, b], width=width, height=height)


class SubExpr(Expr):
    def __init__(self, a: Expr, b: Expr):
        if a.width * a.height == 1:
            # scalar broadcast into all components of b
            width = b.width
            height = b.height
        elif a.width != b.width or a.height != b.height:
            raise RuntimeError(
                f"Invalid shapes ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        else:
            width = a.width
            height = a.height
        super().__init__(args=[a, b], width=width, height=height)


class MulExpr(Expr):
    """
    Element-wise multiplication
    """

    def __init__(self, a: Expr, b: Expr):
        if a.width * a.height == 1:
            # scalar broadcast into all components of b
            width = b.width
            height = b.height
        elif a.width != b.width or a.height != b.height:
            raise RuntimeError(
                f"Invalid shapes ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        else:
            width = a.width
            height = a.height
        super().__init__(args=[a, b], width=width, height=height)


class MatMulExpr(Expr):
    """
    Matrix multiplication
    """

    def __init__(self, a: Expr, b: Expr):
        if a.width != b.height:
            raise RuntimeError(
                f"Invalid shapes ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        super().__init__(args=[a, b], width=b.width, height=a.height)


class DivExpr(Expr):
    """
    Element-wise multiplication
    """

    def __init__(self, a: Expr, b: Expr):
        if b.width * b.height == 1:
            # scalar broadcast into all components of b
            width = a.width
            height = a.height
        elif a.width != b.width or a.height != b.height:
            raise RuntimeError(
                f"Invalid shapes ({a.height}, {a.width}), ({b.height}, {b.width})"
            )
        else:
            width = a.width
            height = a.height
        super().__init__(args=[a, b], width=width, height=height)


class PowExpr(Expr):
    """
    Element-wise exponentiation
    """

    def __init__(self, a: Expr, b: Expr):
        if b.width * b.height != 1:
            raise RuntimeError(f"exponent must a scalar: ({b.height}, {b.width})")
        super().__init__(args=[a, b], width=a.width, height=a.height)


class NegExpr(Expr):
    def __init__(self, a: Expr):
        super().__init__(args=[a], width=a.width, height=a.height)


class GetExpr(Expr):
    """
    operator[]
    """

    def __init__(self, a: Expr, idcs):
        h = range(a.height)
        w = range(a.width)
        get_dim = lambda v: len(v) if type(v) == range else 1
        if type(idcs) == tuple:
            if len(idcs) != 2:
                raise RuntimeError("Expected at most 2 indices")
            h = h[idcs[0]]
            w = w[idcs[1]]
            height = get_dim(h)
            width = get_dim(w)
        elif type(idcs) == slice:
            if a.height == 1:
                # index into width
                w = w[idcs]
            else:
                h = h[idcs]
            height = get_dim(h)
            width = get_dim(w)
        else:
            # single index
            height = 1
            if a.height == 1:
                # index into width
                w = w[idcs]
                width = 1
            else:
                width = a.width
                h = h[idcs]
        super().__init__(args=[a, idcs], width=width, height=height)


class Coeff(Expr):
    """
    General scalar, vector, or matrix coefficient.
    TODO: runtime specified dimensions
    """

    def __init__(self, name: str, width: int = 1, height: int = 1):
        super().__init__(args=[], width=width, height=height)
        self.name = name

    def __str__(self):
        return self.name


class Sqrt(Expr):
    """
    Element-wise sqrt
    """

    def __init__(self, v: Expr):
        super().__init__(args=[v], width=v.width, height=v.height)

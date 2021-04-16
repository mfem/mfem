
#include "complex_coeff.hpp"

void ProductComplexCoefficient::Setup(ComplexCoefficient * A, 
                                      ComplexCoefficient * B)
{
   if (A->real() && B->real()) 
   {
      a = new ProductCoefficient(*A->real(),*B->real()); 
   }
   if (A->imag() && B->imag())
   {
      b = new ProductCoefficient(*A->imag(),*B->imag()); 
   }
   if (a && b)
   {
      own_r = true;
      cr = new SumCoefficient(*a,*b,1.0,-1.0);
   }
   else if (!b)
   {
      cr = a;
   }
   else
   {
      own_r = true;
      cr = new ProductCoefficient(-1.0,*b);
   }

   if (A->real() && B->imag())
   {
      c = new ProductCoefficient(*A->real(),*B->imag());
   }
   if (A->imag() && B->real())
   {
      d = new ProductCoefficient(*A->imag(),*B->real());
   }

   if (c && d)
   {
      own_i = true;
      ci = new SumCoefficient(*c,*d,1.0,1.0);
   }
   else if (!d)
   {
      ci = c;
   }
   else
   {
      ci = d;
   }
}

void ScalarMatrixProductComplexCoefficient::Setup
                                       (ComplexCoefficient * A, 
                                        MatrixComplexCoefficient * B)
{
   if (A->real() && B->real()) 
   {
      a = new ScalarMatrixProductCoefficient(*A->real(),*B->real()); 
   }
   if (A->imag() && B->imag())
   {
      b = new ScalarMatrixProductCoefficient(*A->imag(),*B->imag()); 
   }
   if (a && b)
   {
      own_r = true;
      cr = new MatrixSumCoefficient(*a,*b,1.0,-1.0);
   }
   else if (!b)
   {
      cr = a;
   }
   else
   {
      own_r = true;
      cr = new ScalarMatrixProductCoefficient(-1.0,*b);
   }

   if (A->real() && B->imag())
   {
      c = new ScalarMatrixProductCoefficient(*A->real(),*B->imag());
   }
   if (A->imag() && B->real())
   {
      d = new ScalarMatrixProductCoefficient(*A->imag(),*B->real());
   }

   if (c && d)
   {
      own_i = true;
      ci = new MatrixSumCoefficient(*c,*d,1.0,1.0);
   }
   else if (!d)
   {
      ci = c;
   }
   else
   {
      ci = d;
   }
}



MatrixMatrixProductComplexCoefficient::MatrixMatrixProductComplexCoefficient
                                       (MatrixComplexCoefficient * A, 
                                        MatrixComplexCoefficient * B)
{
   if (A->real() && B->real()) 
   {
      a = new MatrixMatrixProductCoefficient(*A->real(),*B->real()); 
   }
   if (A->imag() && B->imag())
   {
      b = new MatrixMatrixProductCoefficient(*A->imag(),*B->imag()); 
   }
   if (a && b)
   {
      own_r = true;
      cr = new MatrixSumCoefficient(*a,*b,1.0,-1.0);
   }
   else if (!b)
   {
      cr = a;
   }
   else
   {
      own_r = true;
      cr = new ScalarMatrixProductCoefficient(-1.0,*b);
   }

   if (A->real() && B->imag())
   {
      c = new MatrixMatrixProductCoefficient(*A->real(),*B->imag());
   }
   if (A->imag() && B->real())
   {
      d = new MatrixMatrixProductCoefficient(*A->imag(),*B->real());
   }

   if (c && d)
   {
      own_i = true;
      ci = new MatrixSumCoefficient(*c,*d,1.0,1.0);
   }
   else if (!d)
   {
      ci = c;
   }
   else
   {
      ci = d;
   }
}
#include "lh_utils.hpp"

real_t delta = 0.01; 
real_t a0 = -1.0;   
real_t a1 = 5.0;    

void bfunc(const Vector &x, Vector &b)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   int dim = x.Size();
   b.SetSize(dim); b = 0.0;
   b(0) = -x(1) / r;
   b(1) =  x(0) / r;
   if (dim == 3) b(2) = 0.0;
}

void bcrossb(const Vector &x, DenseMatrix &bb)
{
   Vector b;
   bfunc(x, b);
   bb.SetSize(b.Size());
   MultVVt(b, bb);
}

std::complex<real_t> pfunc(const Vector &x)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   return std::complex<real_t>(a0 + a1 *(r-0.9), delta);
}

std::complex<real_t> sfunc(const Vector &x)
{
   return std::complex<real_t>(1.0, delta);
}

real_t pfunc_r(const Vector &x) { return pfunc(x).real(); }
real_t pfunc_i(const Vector &x) { return pfunc(x).imag(); }

real_t sfunc_r(const Vector &x) { return sfunc(x).real(); }
real_t sfunc_i(const Vector &x) { return sfunc(x).imag(); }


void epsilon_func_r(const Vector &x, DenseMatrix &eps)
{
   std::complex<real_t> p = pfunc(x);
   std::complex<real_t> s = sfunc(x);
   DenseMatrix B;
   bcrossb(x, B);
   int dim = x.Size();
   eps.SetSize(dim);

   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         if (i == j)
         {
            eps(i, j) = (s * (1.0 - B(i, j)) + p * B(i, j)).real();
         }
         else
         {
            eps(i, j) = ((p - s) * B(i, j)).real();
         }
      }
   }
}

void epsilon_func_i(const Vector &x, DenseMatrix &eps)
{
   int dim = x.Size();
   eps.SetSize(dim);

   std::complex<real_t> p = pfunc(x);
   std::complex<real_t> s = sfunc(x);
   DenseMatrix B;
   bcrossb(x, B);

   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         if (i == j)
         {
            eps(i, j) = (s * (1.0 - B(i, j)) + p * B(i, j)).imag();
         }
         else
         {
            eps(i, j) = ((p - s) * B(i, j)).imag();
         }
      }
   }
}





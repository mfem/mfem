#include "legendre.hpp"

namespace mfem
{

real_t Gibbs::operator()(const Vector &x) const
{
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      sum += x[i] > tol ? x[i] * std::log(x[i]) : 0.0;
   }
   return sum;
}
void Gibbs::grad(const Vector &x, Vector &g) const
{
   g.SetSize(x.Size());
   for (int i=0; i<x.Size(); i++)
   {
      g[i] = x[i] > tol ? std::log(x[i]) + 1.0 : std::log(tol) + 1.0;
   }
}

void Gibbs::gradinv(const Vector &x, Vector &invg) const
{
   real_t maxval = x.Max();
   invg.SetSize(x.Size());
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      invg[i] = std::exp(x[i] - maxval);
      sum += invg[i];
   }
   invg /= sum;
}

void Gibbs::hessinv(const Vector &x, DenseMatrix &H) const
{
   y.SetSize(x.Size());
   gradinv(x, y);
   H.Diag(y.GetData(), y.Size());
   AddMult_a_VVt(-1.0, y, H);
}


} // namespace mfem

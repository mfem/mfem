#ifndef MFEM_ANDERSON
#define MFEM_ANDERSON

#include "mfem.hpp"

namespace mfem
{
void dispVec(Vector &x)
{
   out << "[";
   for (auto val : x) { out << val << ", "; }
   out << "]" << std::endl;

}
void dispMat(DenseMatrix &mat)
{
   Vector c;
   out << "[";
   for (int i=0; i<mat.Width(); i++)
   {
      mat.GetColumnReference(i, c);
      dispVec(c);
   }
   out << "]" << std::endl;
}
class AndersonAccelerator : public Operator
{
public:
   AndersonAccelerator(const int m, const int siz): Operator(siz), m(m), k(0),
      DRk(siz, m), DXk(siz, m), rk(siz), xk(siz)
   {
      DRk = 0.0; DXk = 0.0; rk = 0.0; xk = 0.0;
   }
   virtual void Mult(const Vector &x, Vector &y) const
   {
      const int mk = std::min(m, k);
      RegisterSequence(x, y);
      // Normal equation

      DenseMatrix N(mk, mk);
      DenseMatrix DRk_view(DRk.GetData(), width, mk);
      DenseMatrix DXk_view(DXk.GetData(), width, mk);
      MultAtB(DRk_view, DRk_view, N);

      Vector rhs(mk);
      DRk_view.MultTranspose(rk, rhs);

      // α = N⁻¹1 / (1ᵀN⁻¹1)
      DenseMatrixInverse invN(&N, true);
      invN.Factor();
      Vector gamma(mk);
      invN.Mult(rhs, gamma);

      // y = Gα
      DXk_view.AddMult_a(1.0, gamma, y);
      DRk_view.AddMult_a(1.0, gamma, y);
   }
   void RegisterSequence(const Vector &x, const Vector &y) const
   {
      if (k == 0)
      {
         xk = x;
         rk = y;
         rk -= x;
      }
      else
      {
         const int idx = (k - 1) % std::min(m, k);
         Vector drk, dxk;

         DRk.GetColumnReference(idx, drk);
         drk = rk;
         drk.Neg();
         rk = y;
         rk -= x;
         drk += rk;

         DXk.GetColumnReference(idx, dxk);
         dxk = xk;
         dxk.Neg();
         xk = x;
         dxk += xk;
      }
      k++;
   }
protected:
   int m;
   Vector one;
   mutable int k;
   mutable DenseMatrix DXk; // Differences of inputs, x_i - x_{i-1}
   mutable DenseMatrix
   DRk; // Differences of remainder, (g(x_i) - x_i) - (g(x_{i-1}) - x_{i-1})
   mutable Vector rk; // remainder, g(x_i) - x_i
   mutable Vector xk; // input, x_i
private:
};
}
#endif

#include "functional.hpp"

namespace mfem
{
QuadraticFunctional::QuadraticFunctional(const Operator *A,
                                         const Vector *b, const real_t beta, const real_t c)
   : Functional(A ? A->Width() : 0)
   , A(A), beta(beta), b(b), c(c)
   , aux(A ? A->Width() : 0)
{
   MFEM_VERIFY(A != nullptr,
               "QuadraticFunctional: A must not be nullptr. "
               << "Use QuadraticFunctional() constructor to create an empty Quadratic functional.");
   MFEM_VERIFY(A->Width() == A->Height(),
               "QuadraticFunctional: A must be a square operator.");
   MFEM_VERIFY(b == nullptr || A->Width() == b->Size(),
               "QuadraticFunctional: A and b must have compatible sizes");
}

void QuadraticFunctional::SetOperator(const Operator &A)
{
   MFEM_VERIFY(A.Width() == A.Height(),
               "QuadraticFunctional: A must be a square operator.");
   this->A = &A;
   width = A.Width();
   aux.SetSize(width);
}

void QuadraticFunctional::SetVector(const Vector &b, const real_t beta)
{
   MFEM_VERIFY(A != nullptr && A->Width() == b.Size(),
               "QuadraticFunctional: A and b must have compatible sizes.");
   this->b = &b;
   this->beta = beta;
}
void QuadraticFunctional::SetConstant(real_t c) { this->c = c; }

void QuadraticFunctional::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(A != nullptr, "QuadraticFunctional: A must not be nullptr");
   A->Mult(x, aux); // aux = A(x)
   if (b != nullptr) { aux.Add(2.0*beta, *b); } // aux = A(x) + 2*beta*b
   real_t result = 0.0;
#ifdef MFEM_USE_MPI
   if (IsParallel())
   {
      result = InnerProduct(GetComm(), x, aux); // result = <A(x) + 2*beta*b, x>
   }
   else
   {
      result = InnerProduct(x, aux); // result = <A(x) + 2*beta*b, x>
   }
#else
   result = InnerProduct(x, aux); // result  = <A(x) + 2*beta*b, x>
#endif
   y.SetSize(1);
   y[0] = result*0.5 + c; // y = 0.5 * <A(x) + 2*beta*b, x> + c
}

void QuadraticFunctional::EvalGradient(const Vector &x,
                                       Vector &y) const
{
   y.SetSize(A->Width());
   A->Mult(x, y);
   if (b) { y.Add(beta,*b); }
}

} // namespace mfem

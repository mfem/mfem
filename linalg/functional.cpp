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


void SharedFunctional::SharePoint(SharedFunctional &viewer)
{
   if (processedX_owner) // If this functional is a viewer,
   {
      if (processedX_owner == viewer.processedX_owner)
      {
         // and the other functional is also a viewer of the same processed point,
         return; // do nothing
      }
      MFEM_VERIFY(viewer.processedX_owner == nullptr,
                  "SharedFunctional::SharePoint(): The other viewer already has a processed point owner. Cannot share the point.");
      processedX_owner->SharePoint(viewer); // Share the owner
   }
   else // if this functional is the owner of the processed point
   {
      viewer.processedX_owner = this;
      processedX_viewers.push_back(&viewer);
      viewer.multCached = viewer.gradCached = viewer.hessianCached =
            false; // invalidate cached values
      viewer.ShallowCopyProcessedX(*this);
   }
}

void SharedFunctional::EnableCache(bool enable)
{
   cache_enabled = enable;
   multCached = gradCached = false;
   current_grad_value.SetSize(Width());
}

void SharedFunctional::Update(const Vector &x) const
{
   if (processedX_owner != nullptr)
   {
      MFEM_WARNING("SharedFunctional::Update() called on a functional that does not own the processed point."
                   << " This is not recommended, as it may lead to incorrect caching behavior."
                   << " This will update the owner's evaluation point and invalidate cached values of owner and its viewers."
                  );
   }
   ProcessX(x);
   multCached = gradCached = hessianCached = false; // invalidate cached values
   for (auto &viewer : processedX_viewers)
   {
      // invalidate the cached values in the viewers
      viewer->multCached = viewer->gradCached = viewer->hessianCached = false;
   }
}

void SharedFunctional::Mult(const Vector &dummy, Vector &y) const
{
   if (multCached) { y = current_mult_value; return; }
   y.SetSize(1);
   if (cache_enabled)
   {
      MultCurrent(y);
      current_mult_value = y[0];
      multCached = true;
   }
   else
   {
      MultCurrent(y);
   }
}

void SharedFunctional::EvalGradient(const Vector &dummy, Vector &y) const
{
   if (gradCached) { y = current_grad_value; return; }

   if (cache_enabled)
   {
      current_grad_value.SetSize(Width());
      EvalGradientCurrent(current_grad_value);
      y = current_grad_value;
      gradCached = true;
      return;
   }
   else
   {
      y.SetSize(Width());
      EvalGradientCurrent(y);
   }
}

void SharedFunctional::ShallowCopyProcessedX(SharedFunctional &owner)
{
   int status;
   char *othername = abi::__cxa_demangle(typeid(owner).name(), nullptr, nullptr,
                                         &status);
   char *thisname = abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr,
                                        &status);
   MFEM_ABORT(thisname << "::ShallowCopyProcessedX() is not implemented for "
              << othername << ".");
}


} // namespace mfem

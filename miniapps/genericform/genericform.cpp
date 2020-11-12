#include "genericform.hpp"

namespace mfem
{
GenericForm::GenericForm(FiniteElementSpace *f)
   : Operator(f->GetTrueVSize()), fes(f), P(f->GetProlongationMatrix())
{
   G = fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_ASSERT(G, "Some GetElementRestriction error");
   x_local.SetSize(G->Height(), Device::GetMemoryType());
   y_local.SetSize(G->Height(), Device::GetMemoryType());
}

const Vector &GenericForm::Prolongate(const Vector &x) const
{
   MFEM_VERIFY(x.Size() == Width(), "invalid input Vector size");
   if (P)
   {
      z1.SetSize(P->Height());
      P->Mult(x, z1);
      return z1;
   }
   return x;
}

void GenericForm::Mult(const Vector &x, Vector &y) const
{
   const Vector &px = Prolongate(x);

   G->Mult(px, x_local);
   for (int i = 0; i < domain_integrators.Size(); ++i)
   {
      domain_integrators[i]->Apply(x_local, y_local);
   }
   G->MultTranspose(y_local, y);
}
} // namespace mfem
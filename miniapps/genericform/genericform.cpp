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

void GenericForm::Mult(const Vector &x, Vector &y) const
{
   px.SetSize(P->Height());
   py.SetSize(P->Height());

   P->Mult(x, px);
   G->Mult(px, x_local);

   y_local = 0.0;
   for (int i = 0; i < domain_integrators.Size(); ++i)
   {
      // y += F(x)
      domain_integrators[i]->Apply(x_local, y_local);
   }
   G->MultTranspose(y_local, py);
   P->MultTranspose(py, y);

   y.HostReadWrite();
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = 0.0;
   }
}
} // namespace mfem
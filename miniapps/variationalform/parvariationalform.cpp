#include "parvariationalform.hpp"
#include "qfuncintegrator.hpp"

namespace mfem
{
ParVariationalForm::ParVariationalForm(ParFiniteElementSpace *f)
   : Operator(f->GetTrueVSize()), fes(f), P(f->GetProlongationMatrix()),
     grad(*this)
{
   G = fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_ASSERT(G, "Some GetElementRestriction error");
   x_local.SetSize(G->Height(), Device::GetMemoryType());
   v_local.SetSize(G->Height(), Device::GetMemoryType());
   y_local.SetSize(G->Height(), Device::GetMemoryType());
}

void ParVariationalForm::Mult(const Vector &x, Vector &y) const
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

void ParVariationalForm::GradientMult(const Vector &v, Vector &y) const
{
   px.SetSize(P->Height());
   py.SetSize(P->Height());
   pv.SetSize(P->Height());

   P->Mult(v, pv);
   G->Mult(pv, v_local);

   P->Mult(x_lin, px);
   G->Mult(px, x_local);

   y_local = 0.0;
   for (int i = 0; i < domain_integrators.Size(); ++i)
   {
      // y += dF(x)/dx * v
      if (is_linear)
      {
         // take care of RHS
         // domain_integrators[i]->Apply(v_local, y_local);
      }
      domain_integrators[i]->ApplyGradient(x_local, v_local, y_local);
   }

   G->MultTranspose(y_local, py);
   P->MultTranspose(py, y);

   y.HostReadWrite();
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = v(ess_tdof_list[i]);
   }
}

Operator &ParVariationalForm::GetGradient(const Vector &x) const
{
   x_lin = x;
   return grad;
}

HypreParMatrix *ParVariationalForm::GetGradientMatrix(const Vector &x)
{
   delete gradient_matrix;
   gradient_matrix = new HypreParMatrix;
   return gradient_matrix;
}

void ParVariationalForm::SetEssentialBC(const Array<int> &ess_attr)
{
   fes->GetEssentialTrueDofs(ess_attr, ess_tdof_list);
}

} // namespace mfem
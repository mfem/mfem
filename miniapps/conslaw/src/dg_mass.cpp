#include "dg_mass.hpp"

namespace mfem
{
namespace dg
{

Mass::Mass(const PartialAssembly *pa_)
   : pa(pa_),
     fes(pa->GetFES())
{
   // Precompute mass matrix
   // For now assume constant number of DOFs per element (i.e. uniform p)
   const int ndof = fes->GetFE(0)->GetDof();
   M.SetSize(ndof, ndof, fes->GetNE());

   // Assemble each local mass matrix and insert into the dense tensor
   MassIntegrator mi;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      mi.AssembleElementMatrix(*fes->GetFE(i),
                               *fes->GetElementTransformation(i),
                               M(i));
   }
}

const PartialAssembly* Mass::GetPA() const
{
   return pa;
}

void Mass::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   const int ndof = fes->GetFE(0)->GetDof();
   DenseMatrix xel, yel;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      int nc = vdofs.Size()/ndof;
      xel.SetSize(ndof, nc);
      yel.SetSize(ndof, nc);
      x.GetSubVector(vdofs, xel.Data());
      mfem::Mult(M(i), xel, yel);
      y.SetSubVector(vdofs, yel.Data());
   }
}

MassInverse::MassInverse(const Mass *mass_)
   : mass(mass_),
     fes(mass->fes)
{
   // For now assume constant number of DOFs per element (i.e. uniform p)
   const int ndof = fes->GetFE(0)->GetDof();
   Minv.SetSize(ndof, ndof, fes->GetNE());

   // Extract the local mass matrices and then invert, inserting the
   // result into the Minv dense tensor
   DenseMatrix Me(ndof);
   DenseMatrixInverse Me_inv(&Me);
   for (int i = 0; i < fes->GetNE(); i++)
   {
      Me = mass->M(i);
      Me_inv.Factor();
      Me_inv.GetInverseMatrix(Minv(i));
   }
}

void MassInverse::Mult(const Vector &x, Vector &y) const
{
   // y must already be allocated/sized here (is that correct?)
   Array<int> vdofs;
   const int ndof = fes->GetFE(0)->GetDof();
   DenseMatrix xel, yel;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      int nc = vdofs.Size()/ndof;
      xel.SetSize(ndof, nc);
      yel.SetSize(ndof, nc);
      x.GetSubVector(vdofs, xel.Data());
      mfem::Mult(Minv(i), xel, yel);
      y.SetSubVector(vdofs, yel.Data());
   }
}

} // namespace dg
} // namespace mfem
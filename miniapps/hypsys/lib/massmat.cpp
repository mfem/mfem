#include "massmat.hpp"

MassMatrixDG::MassMatrixDG(const FiniteElementSpace *fes_) : fes(fes_)
{
   const int nd = fes->GetFE(0)->GetDof();
   M.SetSize(nd, nd, fes->GetNE());
   MassIntegrator mi;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      mi.AssembleElementMatrix(*fes->GetFE(e),
                               *fes->GetElementTransformation(e),
                               M(e));
   }
}

void MassMatrixDG::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   const int nd = fes->GetFE(0)->GetDof();
   DenseMatrix xel, yel;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      fes->GetElementVDofs(e, vdofs);
      int nc = vdofs.Size()/nd;
      xel.SetSize(nd, nc);
      yel.SetSize(nd, nc);
      x.GetSubVector(vdofs, xel.Data());
      mfem::Mult(M(e), xel, yel);
      y.SetSubVector(vdofs, yel.Data());
   }
}

InverseMassMatrixDG::InverseMassMatrixDG(const MassMatrixDG *mass_)
   : mass(mass_),
     fes(mass->fes)
{
   const int nd = fes->GetFE(0)->GetDof();
   Minv.SetSize(nd, nd, fes->GetNE());

   DenseMatrix Me(nd);
   DenseMatrixInverse MeInv(&Me);
   for (int i = 0; i < fes->GetNE(); i++)
   {
      Me = mass->M(i);
      MeInv.Factor();
      MeInv.GetInverseMatrix(Minv(i));
   }
}

void InverseMassMatrixDG::Mult(const Vector &x, Vector &y) const
{
   Array<int> vdofs;
   const int nd = fes->GetFE(0)->GetDof();
   DenseMatrix xel, yel;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      int nc = vdofs.Size()/nd;
      xel.SetSize(nd, nc);
      yel.SetSize(nd, nc);
      x.GetSubVector(vdofs, xel.Data());
      mfem::Mult(Minv(i), xel, yel);
      y.SetSubVector(vdofs, yel.Data());
   }
}

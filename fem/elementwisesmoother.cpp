// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "fem.hpp"

namespace mfem
{

ElementWiseSmoother::ElementWiseSmoother(const mfem::FiniteElementSpace& fespace)
   :
   mfem::Solver(fespace.GetVSize()),
   fespace_(fespace)
{
}

void ElementWiseSmoother::Mult(const Vector& b, Vector& x) const
{
   Vector local_residual;
   Vector local_correction;
   Vector b_local;
   Array<int> local_dofs;

   if (!iterative_mode)
   {
      x = 0.0;
   }

   for (int e = 0; e < fespace_.GetNE(); ++e)
   {
      fespace_.GetElementDofs(e, local_dofs);
      b.GetSubVector(local_dofs, b_local);
      local_correction.SetSize(b_local.Size());
      ElementResidual(e, b_local, x, local_residual);

      LocalSmoother(e, local_residual, local_correction);

      x.AddElementVector(local_dofs, local_correction);
   }
}

void ElementWiseSmoother::ElementResidual(int e, const Vector& b_local,
                                          const Vector& x,
                                          Vector& r_local) const
{
   r_local.SetSize(b_local.Size());
   r_local = 0.0;

   GetElementFromMatVec(e, x, r_local);
   r_local -= b_local;
   r_local *= -1.0;
}

ElementWiseJacobi::ElementWiseJacobi(const mfem::FiniteElementSpace& fespace,
                                     mfem::BilinearForm& aform,
                                     const mfem::Vector& global_diag,
                                     double scale)
   :
   ElementWiseSmoother(fespace),
   aform_(aform),
   global_diag_(global_diag),
   scale_(scale)
{
   const Table& el_dof = fespace_.GetElementToDofTable();
   Table dof_el;
   mfem::Transpose(el_dof, dof_el);
   mfem::Mult(el_dof, dof_el, el_to_el_);
}

// x is *global*, y is local to the element
void ElementWiseJacobi::GetElementFromMatVec(int element, const mfem::Vector& x,
                                             mfem::Vector& y_element) const
{
   Array<int> neighbors;
   el_to_el_.GetRow(element, neighbors);
   Array<int> row_dofs;
   Array<int> col_dofs;
   fespace_.GetElementDofs(element, row_dofs);
   Vector x_local;
   Vector z_element;
   DenseMatrix elmat;

   y_element.SetSize(row_dofs.Size());
   z_element.SetSize(row_dofs.Size());
   y_element = 0.0;

   for (int i = 0; i < neighbors.Size(); ++i)
   {
      int ne = neighbors[i];
      fespace_.GetElementDofs(ne, col_dofs);

      z_element = 0.0;
      aform_.ElementMatrixMult(ne, x, z_element);

      // okay, this next section seems very inefficient
      // (could probably store and precompute some kind of map?)
      for (int j = 0; j < row_dofs.Size(); ++j)
      {
         const int rd = row_dofs[j];
         for (int k = 0; k < col_dofs.Size(); ++k)
         {
            const int cd = col_dofs[k];
            if (rd == cd)
            {
               y_element[j] += z_element[k];
               break;
            }
         }
      }
   }
}

void ElementWiseJacobi::LocalSmoother(int e, const Vector& in, Vector& out) const
{
   DenseMatrix elmat;
   Array<int> local_dofs;
   fespace_.GetElementDofs(e, local_dofs);
   for (int i = 0; i < in.Size(); ++i)
   {
      out[i] = (scale_ / global_diag_(local_dofs[i])) * in[i];
   }
}

}

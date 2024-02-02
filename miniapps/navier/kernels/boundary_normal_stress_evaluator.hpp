// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "general/forall.hpp"

namespace mfem
{
namespace navier
{

void BoundaryNormalStressEvaluator(ParGridFunction &u_gf,
                                   ParGridFunction &p_gf,
                                   ParGridFunction &nu_gf,
                                   Array<int> &marker,
                                   IntegrationRule &ir_face,
                                   ParGridFunction &sigmaN_gf,
                                   const double density = 1.0)
{
   const auto &u_fes = *u_gf.ParFESpace();
   // const auto &p_fes = *p_gf.ParFESpace();
   // const auto &nu_fes = *nu_gf.ParFESpace();
   const auto &sigmaN_fes = *u_gf.ParFESpace();
   const auto &mesh = *u_fes.GetParMesh();
   const auto dim = mesh.Dimension();
   sigmaN_gf = 0.;

   Array<int> sigmaN_vdofs;

   double nu;
   Vector sigmaN, nor(dim);
   DenseMatrix dshape, dudxi, dudx, A(dim, dim);

   for (int be = 0; be < u_fes.GetNBE(); be++)
   {
      const int bdr_el_attr = mesh.GetBdrAttribute(be);
      if (marker[bdr_el_attr-1] == 0)
      {
         continue;
      }

      sigmaN_fes.GetBdrElementVDofs(be, sigmaN_vdofs);
      sigmaN_gf.GetSubVector(sigmaN_vdofs, sigmaN);

      const FiniteElement &bdr_el = *u_fes.GetBE(be);
      ElementTransformation &Tr = *u_fes.GetBdrElementTransformation(be);

      const int ndofs = bdr_el.GetDof();
      for (int dof = 0; dof < ndofs; ++dof)
      {
         const IntegrationPoint &ip = bdr_el.GetNodes().IntPoint(dof);
         Tr.SetIntPoint(&ip);
         CalcOrtho(Tr.Jacobian(), nor);

         const double scale = nor.Norml2();
         nor /= scale;

         u_gf.GetVectorGradient(Tr, dudx);
         auto p = p_gf.GetValue(Tr, ip);
         nu = nu_gf.GetValue(Tr, ip);

         // (-p * I + nu (grad(u) + grad(u)^T))
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               A(i, j) = -p * (i == j) + density * nu * (dudx(i, j) + dudx(j, i));
               // A(i, j) = -p * (i == j);
               // A(i, j) = nu * (dudx(i, j) + dudx(j, i));
            }
         }

         for (int i = 0; i < dim; i++)
         {
            double s = 0.0;
            for (int j = 0; j < dim; j++)
            {
               s += A(i, j) * nor(j);
            }
            sigmaN(ndofs * i + dof) = s;
         }
      }
      sigmaN_gf.SetSubVector(sigmaN_vdofs, sigmaN);
   }
   Vector true_sigmaN;
   sigmaN_gf.GetTrueDofs(true_sigmaN);
   sigmaN_gf.SetFromTrueDofs(true_sigmaN);
}

}
}
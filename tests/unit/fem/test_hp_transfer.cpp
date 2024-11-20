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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;
namespace hptransfer_test
{

int order=1;

double u(const Vector & x)
{
   return pow(x.Sum(),order);
}

void vecu(const Vector & x, Vector & U)
{
   for (int i = 0; i<x.Size(); i++)
   {
      U[i] = pow(x[i], order);
   }
}

void RandomPRefinement(FiniteElementSpace & fes)
{
   Mesh *mesh = fes.GetMesh();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      if ((double) rand() / RAND_MAX < 0.5)
      {
         const int eorder = fes.GetElementOrder(i);
         fes.SetElementOrder(i,eorder+1);
      }
   }
   fes.Update(false);
}

/* This function randomly selects elements to be de-refined and sets the
   order of the elements that share the same parent to their minimum */
void PreprocessRandomDerefinement(FiniteElementSpace & fes, Array<int> &drefs,
                                  double prob=0.5)
{
   Mesh * mesh = fes.GetMesh();
   const Table & dereftable = mesh->ncmesh->GetDerefinementTable();
   int dref = dereftable.Size();
   for (int i = 0; i < dref; i++)
   {
      if ((double) rand() / RAND_MAX < prob)
      {
         drefs.Append(i);
      }
   }

   // Go through the possible derefinements and set the orders to minimum
   Array<int> row;
   for (int i = 0; i<drefs.Size(); i++)
   {
      dereftable.GetRow(drefs[i], row);
      int minorder = 100;
      for (int j = 0; j<row.Size(); j++)
      {
         minorder = std::min(minorder, fes.GetElementOrder(row[j]));
      }
      // set the min order
      for (int j = 0; j<row.Size(); j++)
      {
         fes.SetElementOrder(row[j],minorder);
      }
   }
   fes.Update(false);
}

void Derefine(Mesh &mesh, const Array<int> &drefs)
{
   const Table & dereftable = mesh.ncmesh->GetDerefinementTable();

   Array<int> row;
   Vector errors(mesh.GetNE()); errors = infinity();
   for (int i = 0; i<drefs.Size(); i++)
   {
      dereftable.GetRow(drefs[i], row);
      for (int j = 0; j<row.Size(); j++)
      {
         errors[row[j]] = 0.0;
      }
   }
   mesh.DerefineByError(errors,1.0);
}

enum class Space {H1, L2, VectorH1, VectorL2};

TEST_CASE("hpTransfer", "[hpTransfer]")
{
   auto space   = GENERATE(Space::H1, Space::L2, Space::VectorH1, Space::VectorL2);
   int dim      = GENERATE(2,3);
   auto simplex = GENERATE(false, true);
   order        = GENERATE(1,2);
   auto relax_conformity = GENERATE(false, true);

   /* No need to distinguish between relaxed and full conformity in the DG case*/
   if ((space == Space::L2 || space == Space::VectorL2) && relax_conformity) { return; }

   constexpr int ne = 3;

   CAPTURE(space, dim, simplex, order, relax_conformity);

   Mesh mesh;
   if (dim == 2)
   {
      Element::Type type = simplex ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
   }
   else
   {
      Element::Type type = simplex ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
   }
   mesh.EnsureNCMesh(true);

   // 1. Set up initial state by randomly h- and p- refinement
   mesh.RandomRefinement(0.5);

   FiniteElementCollection * fec = nullptr;
   if (space == Space::H1 || space == Space::VectorH1)
   {
      fec = new H1_FECollection(order, dim);
   }
   else
   {
      fec = new L2_FECollection(order, dim);
   }

   int dimc = (space<=Space::L2) ? 1 : dim;
   FiniteElementSpace fes(&mesh, fec, dimc);
   fes.SetRelaxedHpConformity(relax_conformity);
   RandomPRefinement(fes);

   // 2. Set up a GridFunction on the initial hp-mesh
   FunctionCoefficient f(u);
   VectorFunctionCoefficient vf(dim,vecu);
   GridFunction gf(&fes); gf = 0.0;
   if (space<=Space::L2)
   {
      gf.ProjectCoefficient(f);
   }
   else
   {
      gf.ProjectCoefficient(vf);
   }

   // 3. Randomly h-refine the mesh and transfer the GridFunction
   mesh.RandomRefinement(0.5);
   fes.Update();
   gf.Update();

   GridFunction err_gf(&fes);
   if (space<=Space::L2)
   {
      err_gf.ProjectCoefficient(f);
   }
   else
   {
      err_gf.ProjectCoefficient(vf);
   }
   err_gf-= gf;

   if (fes.GetHpRestrictionMatrix())
   {
      Vector tmp0(fes.GetHpRestrictionMatrix()->Height());
      fes.GetHpRestrictionMatrix()->Mult(err_gf,tmp0);
      fes.GetProlongationMatrix()->Mult(tmp0,err_gf);
   }

   // 3a. Check if the prolonged GridFunction to the h-refined
   //     mesh exactly reproduces the polynomial GridFunction
   REQUIRE(err_gf.Norml2() < 1e-11);

   // 4. Randomly p-refine the mesh and transfer the GridFunction
   Mesh cmesh(mesh);
   FiniteElementSpace cfes(&cmesh, fec, dimc);
   cfes.SetRelaxedHpConformity(relax_conformity);
   for (int i = 0; i<cmesh.GetNE(); i++)
   {
      cfes.SetElementOrder(i,fes.GetElementOrder(i));
   }
   cfes.Update(false);

   RandomPRefinement(fes);
   PRefinementTransferOperator T(cfes, fes);

   GridFunction hpgf(&fes);
   T.Mult(gf,hpgf);

   err_gf.SetSpace(&fes);
   if (space<=Space::L2)
   {
      err_gf.ProjectCoefficient(f);
   }
   else
   {
      err_gf.ProjectCoefficient(vf);
   }
   err_gf-= hpgf;

   if (fes.GetHpRestrictionMatrix())
   {
      Vector tmp(fes.GetHpRestrictionMatrix()->Height());
      fes.GetHpRestrictionMatrix()->Mult(err_gf,tmp);
      fes.GetProlongationMatrix()->Mult(tmp,err_gf);
   }

   // 4a. Check if the prolonged GridFunction to the p-refined
   //     mesh exactly reproduces the polynomial GridFunction
   REQUIRE(err_gf.Norml2() < 1e-11);

   // 5. Before randomly de-refining the mesh ensure that the elements
   //    (of the same parent) that are going to be de-refined
   //    have the same order
   Mesh fmesh(mesh);
   FiniteElementSpace ffes(&fmesh, fec, dimc);
   ffes.SetRelaxedHpConformity(relax_conformity);
   for (int i = 0; i<fmesh.GetNE(); i++)
   {
      ffes.SetElementOrder(i,fes.GetElementOrder(i));
   }
   ffes.Update(false);

   Array<int> drefs;
   // lower the order of the children to their minimum
   PreprocessRandomDerefinement(fes, drefs);
   PRefinementTransferOperator T2(ffes, fes);
   gf.SetSpace(&fes);
   T2.Mult(hpgf, gf);

   err_gf.SetSpace(&fes);
   if (space<=Space::L2)
   {
      err_gf.ProjectCoefficient(f);
   }
   else
   {
      err_gf.ProjectCoefficient(vf);
   }
   err_gf-= gf;

   if (fes.GetHpRestrictionMatrix())
   {
      Vector temp(fes.GetHpRestrictionMatrix()->Height());
      fes.GetHpRestrictionMatrix()->Mult(err_gf,temp);
      fes.GetProlongationMatrix()->Mult(temp,err_gf);
   }

   // 5a. Check if the restricted GridFunction to the p-derefined
   //     mesh exactly reproduces the polynomial GridFunction
   REQUIRE(err_gf.Norml2() < 1e-11);

   // 6. De-refine the mesh and transfer the GridFunction
   Derefine(mesh,drefs);

   fes.Update();
   gf.Update();

   err_gf.SetSpace(&fes); err_gf = 0.0;
   if (space<=Space::L2)
   {
      err_gf.ProjectCoefficient(f);
   }
   else
   {
      err_gf.ProjectCoefficient(vf);
   }

   err_gf-= gf;

   if (fes.GetHpRestrictionMatrix())
   {
      Vector temp(fes.GetHpRestrictionMatrix()->Height());
      fes.GetHpRestrictionMatrix()->Mult(err_gf,temp);
      fes.GetProlongationMatrix()->Mult(temp,err_gf);
   }

   // 6a. Check if the restricted GridFunction to the de-refined
   //     mesh exactly reproduces the polynomial GridFunction
   REQUIRE(err_gf.Norml2() < 1e-11);
   delete fec;
}

}

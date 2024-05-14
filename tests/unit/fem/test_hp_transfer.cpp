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

int order;

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
   int maxorder = 0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const int eorder = fes.GetElementOrder(i);
      maxorder = std::max(maxorder,order);
      if ((double) rand() / RAND_MAX < 0.5)
      {
         fes.SetElementOrder(i,eorder+1);
         maxorder = std::max(maxorder,eorder+1);
      }
   }
   fes.Update(false);
}

void PreprocessRandomDerefinment(FiniteElementSpace & fes, Array<int> &drefs,
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

   // Go through the possible drefinments and set the orders to minimum
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

   // Go through the possible drefinments and set the orders to minimum
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
   int dim      = GENERATE(2, 3);
   auto simplex = GENERATE(true, false);
   order        = GENERATE(1, 2);

   int ne = 3;

   CAPTURE(space, dim, simplex, order);

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

   mesh.RandomRefinement(0.5);

   FiniteElementCollection *fec = nullptr;
   switch (space)
   {
      case Space::H1:
      case Space::VectorH1:
         fec = new H1_FECollection(order, dim);
         break;
      case Space::L2:
      case Space::VectorL2:
         fec = new L2_FECollection(order, dim);
         break;
   }

   int dimc = (space<=Space::L2) ? 1 : dim;
   FiniteElementSpace fes(&mesh, fec, dimc);
   fes.SetRelaxedHpConformity();

   RandomPRefinement(fes);

   FunctionCoefficient f(u);
   VectorFunctionCoefficient vf(dim,vecu);
   GridFunction gf(&fes);
   if (space<=Space::L2)
   {
      gf.ProjectCoefficient(f);
   }
   else
   {
      gf.ProjectCoefficient(vf);
   }
   // State 1: hrefined
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

   REQUIRE(err_gf.Norml2() < 1e-11);

   // copy the mesh and the fes to check transfer operator
   Mesh cmesh(mesh);
   FiniteElementSpace cfes(&cmesh, fec, dimc);
   cfes.SetRelaxedHpConformity();
   for (int i = 0; i<cmesh.GetNE(); i++)
   {
      cfes.SetElementOrder(i,fes.GetElementOrder(i));
   }
   cfes.Update(false);

   // State 2: prefined
   RandomPRefinement(fes);
   PRefinementTransferOperator T(cfes, fes);

   GridFunction hpgf(&fes); hpgf = 0.0;
   T.Mult(gf,hpgf);

   GridFunction err_hpgf(&fes);
   if (space<=Space::L2)
   {
      err_hpgf.ProjectCoefficient(f);
   }
   else
   {
      err_hpgf.ProjectCoefficient(vf);
   }

   err_hpgf-= hpgf;

   REQUIRE(err_hpgf.Norml2() < 1e-11);

   Mesh fmesh(mesh);
   FiniteElementSpace ffes(&fmesh, fec, dimc);
   ffes.SetRelaxedHpConformity();
   for (int i = 0; i<fmesh.GetNE(); i++)
   {
      ffes.SetElementOrder(i,fes.GetElementOrder(i));
   }
   ffes.Update(false);

   Array<int> drefs;
   PreprocessRandomDerefinment(fes,
                               drefs); // lower the order of the children to their minimum
   PRefinementTransferOperator T2(ffes, fes);
   GridFunction gf2(&fes);
   T2.Mult(hpgf, gf2);

   GridFunction err_gf2(&fes);

   if (space<=Space::L2)
   {
      err_gf2.ProjectCoefficient(f);
   }
   else
   {
      err_gf2.ProjectCoefficient(vf);
   }

   err_gf2-= gf2;

   REQUIRE(err_gf2.Norml2() < 1e-11);

   Derefine(mesh,drefs);

   fes.Update();
   gf2.Update();

   GridFunction err_gf3(&fes);
   if (space<=Space::L2)
   {
      err_gf3.ProjectCoefficient(f);
   }
   else
   {
      err_gf3.ProjectCoefficient(vf);
   }

   err_gf3-= gf2;

   REQUIRE(err_gf3.Norml2() < 1e-11);

   delete fec;
}

}
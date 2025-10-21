// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

namespace derefine
{

int dimension;
double coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return sin(10.0*(x[0]+x[1]));
   }
   else
   {
      return sin(10.0*(x[0]+x[1]+x[2]));
   }
}

TEST_CASE("Derefine")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int order = 0; order <= 2; ++order)
      {
         for (int map_type = FiniteElement::VALUE; map_type <= FiniteElement::INTEGRAL;
              ++map_type)
         {
            const int ne = 8;
            Mesh mesh;
            if (dimension == 2)
            {
               mesh = Mesh::MakeCartesian2D(
                         ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(
                         ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }

            mesh.EnsureNCMesh();
            mesh.SetCurvature(std::max(order,1), false, dimension, Ordering::byNODES);

            L2_FECollection fec(order, dimension, BasisType::Positive, map_type);

            FiniteElementSpace fespace(&mesh, &fec);

            GridFunction x(&fespace);

            FunctionCoefficient c(coeff);
            x.ProjectCoefficient(c);

            fespace.Update();
            x.Update();

            Array<Refinement> refinements;
            refinements.Append(Refinement(1));
            refinements.Append(Refinement(2));

            int nonconformity_limit = 0; // 0 meaning allow unlimited ratio

            // First refine two elements.
            mesh.GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Now refine one more element and then derefine it, comparing x before and after.
            Vector diff(x);

            refinements.DeleteAll();
            refinements.Append(Refinement(2));
            mesh.GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Derefine by setting 0 error on the fine elements in coarse element 2.
            Table coarse_to_fine_;
            const CoarseFineTransformations &rtrans = mesh.GetRefinementTransforms();
            rtrans.MakeCoarseToFineTable(coarse_to_fine_);
            Array<int> tabrow;

            Vector local_err(mesh.GetNE());
            double threshold = 1.0;
            local_err = 2*threshold;
            coarse_to_fine_.GetRow(2, tabrow);
            for (int j = 0; j < tabrow.Size(); j++) { local_err(tabrow[j]) = 0.0; }
            mesh.DerefineByError(local_err, threshold, 0, 1);

            fespace.Update();
            x.Update();

            diff -= x;
            REQUIRE(diff.Norml2() / x.Norml2() < 1e-11);
         }
      }
   }
}


double integrate(GridFunction* gf)
{
   ConstantCoefficient one(1.0);
   LinearForm lf(gf->FESpace());
   LinearFormIntegrator* lfi = new DomainLFIntegrator(one);
   lf.AddDomainIntegrator(lfi);
   lf.Assemble();
   double integral = lf(*gf);
   return integral;
}

struct PolyCoeff
{
   static int order_;

   static double poly_coeff(const Vector& x)
   {
      int& o{order_};

      double f = 0.0;
      for (int d = 0; d < dimension; d++)
      {
         f += pow(x[d],o);
      }
      return f;
   }
};
int PolyCoeff::order_ = -1;

void test_derefine_L2_element(int order, Element::Type el_type, int basis_type)
{
   Mesh mesh;
   if (dimension == 1)
   {
      mesh = Mesh::MakeCartesian1D(1, 1.0);
   }
   if (dimension == 2)
   {
      mesh = Mesh::MakeCartesian2D(1, 1, el_type, true, 1.0, 1.0);
   }
   if (dimension == 3)
   {
      mesh = Mesh::MakeCartesian3D(1, 1, 1, el_type, true, 1.0, 1.0, 1.0);
   }
   mesh.EnsureNCMesh(true);

   L2_FECollection fec(order, dimension, basis_type);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order+1; // raise order so it isn't exact
   FunctionCoefficient c(PolyCoeff::poly_coeff);

   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // project to get function that isn't exactly representable in the
   // fine space.
   x.ProjectCoefficient(c);

   // deep copy the fine solution
   GridFunction x_fine(x);

   double mass_fine = integrate(&x_fine);

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);
   fespace.Update();
   x.Update();

   Vector coarse_soln_v{x};

   double mass_coarse = integrate(&x);

   // conservation check
   REQUIRE( fabs(mass_fine-mass_coarse) < 1.e-12 );

   // re-refine to get everything on the same grid
   mesh.GeneralRefinement(refinements);
   fespace.Update();
   x.Update();

   // Compute error of coarse L2 projection against fine solution
   GridFunctionCoefficient gfc(&x);

   double err0 = x_fine.ComputeL2Error(gfc);

   // test for local L2 optimality by shifting dofs by +/-epsilon and
   // recomputing error wrt fine solution. maybe there is a more
   // clever way to do this.
   double eps = 1.e-3;

   // limit to max 20 dofs for efficiency in 3D
   int test_ndofs = std::min(coarse_soln_v.Size(), 20);
   for (int i = 0; i < test_ndofs; i++)
   {
      for (int f = -1; f <= 1; f += 2)
      {
         mesh.DerefineByError(local_err, threshold);
         fespace.Update();
         x.Update();
         x = coarse_soln_v;
         x.HostReadWrite();
         x(i) += f*eps;

         mesh.GeneralRefinement(refinements);
         fespace.Update();
         x.Update();

         double err = x_fine.ComputeL2Error(gfc);
         REQUIRE(err > err0);
      }
   }
}

TEST_CASE("AMR Coarsen L2 Element","[AMR][Coarsen][CUDA]")
{
   std::vector<int> orders_1d{0,1,2,3};
   std::vector<int> orders_2d{0,1,2,3};
   std::vector<int> orders_3d{0,1,2};

   std::vector<int> basis_types
   {
      BasisType::Positive,
      BasisType::GaussLegendre,
      BasisType::GaussLobatto};

   std::vector<Element::Type> el_types_1d;
   el_types_1d.push_back(Element::SEGMENT);

   std::vector<Element::Type> el_types_2d;
   el_types_2d.push_back(Element::QUADRILATERAL);
   el_types_2d.push_back(Element::TRIANGLE); // (nonconforming only)

   std::vector<Element::Type> el_types_3d;
   el_types_3d.push_back(Element::HEXAHEDRON);
   el_types_3d.push_back(Element::TETRAHEDRON); // (nonconforming only)
   // el_types_3d.push_back(Element::WEDGE); // derefinement not supported

   dimension = 1;
   for (auto el_type: el_types_1d)
   {
      for (auto order: orders_1d)
      {
         for (auto basis_type: basis_types)
         {
            test_derefine_L2_element(order, el_type, basis_type);
         }
      }
   }

   dimension = 2;
   for (auto el_type: el_types_2d)
   {
      for (auto order: orders_2d)
      {
         for (auto basis_type: basis_types)
         {
            test_derefine_L2_element(order, el_type, basis_type);
         }
      }
   }

   dimension = 3;
   for (auto el_type: el_types_3d)
   {
      for (auto order: orders_3d)
      {
         for (auto basis_type: basis_types)
         {
            test_derefine_L2_element(order, el_type, basis_type);
         }
      }
   }
}

#ifdef MFEM_USE_MPI

void RefineRandomly(ParMesh& pmesh,
                    ParFiniteElementSpace& fespace, ParGridFunction& u)
{
   double freq = 0.3;

   Array<Refinement> refinements;
   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      double a = rand()/double(RAND_MAX);
      if (a < freq)
      {
         refinements.Append(Refinement(k));
      }
   }

   pmesh.GeneralRefinement(refinements);
   fespace.Update();
   u.Update();
}


void CoarsenRandomly(ParMesh& pmesh,
                     ParFiniteElementSpace& fespace, ParGridFunction& u)
{
   double freq = 0.2;

   Vector local_err(pmesh.GetNE());
   local_err = 1.1;
   double threshold = 1.0;

   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      double a = rand()/double(RAND_MAX);
      if (a < freq)
      {
         local_err(k) = 0.0;
      }
   }

   int op = 0; // take min
   int nc_limit = 0;
   pmesh.DerefineByError(local_err, threshold, nc_limit, op);
   fespace.Update();
   u.Update();
}

// stress test parallel correctness by:
// initializing with an exact polynomial
// loop:
//   refine randomly
//   rebalance
//   coarsen randomly
//   rebalance
// representation should remain exact throughout

void stress_parallel_coarsen(int order, Element::Type el_type, int basis_type)
{
   // This is deactivated until we have a way to exactly project onto
   // positive finite elements.
#if 0
   int myid = Mpi::WorldRank();

   Mesh mesh;
   if (dimension == 1)
   {
      mesh = Mesh::MakeCartesian1D(4, 1.0);
   }
   if (dimension == 2)
   {
      mesh = Mesh::MakeCartesian2D(2, 2, el_type, true, 1.0, 1.0);
   }
   if (dimension == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2, el_type, true, 1.0, 1.0, 1.0);
   }
   mesh.EnsureNCMesh(true);

   ParMesh *pmeshp = new ParMesh(MPI_COMM_WORLD, mesh);
   ParMesh& pmesh{*pmeshp};

   L2_FECollection fec(order, dimension, basis_type);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParGridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order; // exact
   FunctionCoefficient c(PolyCoeff::poly_coeff);

   x.ProjectCoefficient(c);

   srand( (unsigned)time( NULL )+myid );

   int total_it = 10;
   for (int it = 0; it < total_it; it++)
   {
      RefineRandomly(pmesh, fespace, x);

      double err = x.ComputeL2Error(c);
      REQUIRE( err < 1.e-12 );

      pmesh.Rebalance();
      fespace.Update();
      x.Update();

      err = x.ComputeL2Error(c);
      REQUIRE( err < 1.e-12 );

      CoarsenRandomly(pmesh, fespace, x);

      err = x.ComputeL2Error(c);
      REQUIRE( err < 1.e-12 );

      pmesh.Rebalance();
      fespace.Update();
      x.Update();

      err = x.ComputeL2Error(c);
      REQUIRE( err < 1.e-12 );
   }
#endif
}

TEST_CASE("Parallel AMR Coarsen Stress Test", "[AMR][Coarsen][Parallel][CUDA]")
{
   std::vector<int> orders_1d{0,1,2,3};
   std::vector<int> orders_2d{0,1,2,3};
   std::vector<int> orders_3d{0,1,2};

   std::vector<int> basis_types
   {
      BasisType::Positive,
      BasisType::GaussLegendre,
      BasisType::GaussLobatto};

   std::vector<Element::Type> el_types_1d;
   el_types_1d.push_back(Element::SEGMENT);

   std::vector<Element::Type> el_types_2d;
   el_types_2d.push_back(Element::QUADRILATERAL);
   // el_types_2d.push_back(Element::TRIANGLE); // derefinement not supported

   std::vector<Element::Type> el_types_3d;
   el_types_3d.push_back(Element::HEXAHEDRON);
   // el_types_3d.push_back(Element::TETRAHEDRON); // derefinement not supported
   // el_types_3d.push_back(Element::WEDGE); // derefinement not supported

   dimension = 1;
   for (auto el_type: el_types_1d)
   {
      for (auto order: orders_1d)
      {
         for (auto basis_type: basis_types)
         {
            stress_parallel_coarsen(order, el_type, basis_type);
         }
      }
   }

   dimension = 2;
   for (auto el_type: el_types_2d)
   {
      for (auto order: orders_2d)
      {
         for (auto basis_type: basis_types)
         {
            stress_parallel_coarsen(order, el_type, basis_type);
         }
      }
   }

   dimension = 3;
   for (auto el_type: el_types_3d)
   {
      for (auto order: orders_3d)
      {
         for (auto basis_type: basis_types)
         {
            stress_parallel_coarsen(order, el_type, basis_type);
         }
      }
   }

}

TEST_CASE("ParDerefine", "[Parallel][CUDA]")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int order = 0; order <= 2; ++order)
      {
         for (int map_type = FiniteElement::VALUE; map_type <= FiniteElement::INTEGRAL;
              ++map_type)
         {
            const int ne = 8;
            Mesh mesh;
            if (dimension == 2)
            {
               mesh = Mesh::MakeCartesian2D(
                         ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(
                         ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }

            mesh.EnsureNCMesh();
            mesh.SetCurvature(std::max(order,1), false, dimension, Ordering::byNODES);

            ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

            L2_FECollection fec(order, dimension, BasisType::Positive, map_type);
            ParFiniteElementSpace fespace(pmesh, &fec);

            ParGridFunction x(&fespace);

            FunctionCoefficient c(coeff);
            x.ProjectCoefficient(c);

            fespace.Update();
            x.Update();

            // Refine two elements on each process and then derefine, comparing
            // x before and after.
            Vector diff(x);

            Array<Refinement> refinements;
            refinements.Append(Refinement(1));
            refinements.Append(Refinement(2));

            int nonconformity_limit = 0; // 0 meaning allow unlimited ratio

            pmesh->GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Derefine by setting 0 error on all fine elements.
            Vector local_err(pmesh->GetNE());
            double threshold = 1.0;
            local_err = 0.0;
            pmesh->DerefineByError(local_err, threshold, 0, 1);

            fespace.Update();
            x.Update();

            diff -= x;
            REQUIRE(diff.Norml2() / x.Norml2() < 1e-11);

            delete pmesh;
         }
      }
   }
}
#endif
}

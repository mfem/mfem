// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//    ---------------------------------------------------------------
//    Geometric multigrid for the clamped biharmonic problem in 2D
//    ---------------------------------------------------------------
//
// Solves the clamped plate problem (D^2 u, D^2 v) = (f, v) with an HCT, reduced
// HCT, Bell or Argyris discretization, preconditioned by geometric multigrid on
// a uniformly refined mesh hierarchy.
//
// The intergrid transfer is interpolation of the degrees of freedom, provided
// by FiniteElement::GetTransferMatrix for each element, so the hierarchy and
// its prolongations come straight from FiniteElementSpaceHierarchy. The HCT and
// Bell spaces are not nested under refinement, and there the transfer is an
// interpolation rather than an embedding.
//
// Sample runs:
//    biharmonic_gmg -e argyris -r 3
//    biharmonic_gmg -e bell -r 3
//    biharmonic_gmg -e rhct -r 4 -m ../../data/square-disc.mesh

#include "mfem.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace mfem;
using namespace std;

namespace
{

unique_ptr<FiniteElementCollection> MakeCollection(const string &element)
{
   if (element == "hct")
   {
      return unique_ptr<FiniteElementCollection>(new HCT_FECollection);
   }
   if (element == "rhct")
   {
      return unique_ptr<FiniteElementCollection>(new ReducedHCT_FECollection);
   }
   if (element == "bell")
   {
      return unique_ptr<FiniteElementCollection>(new BellFECollection);
   }
   if (element == "argyris")
   {
      return unique_ptr<FiniteElementCollection>(new ArgyrisFECollection);
   }
   MFEM_ABORT("Unknown element type '" << element << "'.");
   return nullptr;
}

/// Bending energy on every level, with point Gauss-Seidel smoothing.
class BiharmonicMultigrid : public GeometricMultigrid
{
public:
   BiharmonicMultigrid(FiniteElementSpaceHierarchy &hierarchy,
                       const Array<int> &essential_boundary)
      : GeometricMultigrid(hierarchy, essential_boundary)
   {
      for (int level = 0; level < hierarchy.GetNumLevels(); level++)
      {
         FiniteElementSpace &fes = hierarchy.GetFESpaceAtLevel(level);
         BilinearForm *form = new BilinearForm(&fes);
         form->AddDomainIntegrator(new HessianIntegrator);
         form->Assemble();
         bfs.Append(form);

         OperatorPtr system;
         system.SetType(Operator::MFEM_SPARSEMAT);
         form->FormSystemMatrix(*essentialTrueDofs[level], system);
         system.SetOperatorOwner(false);
         if (level > 0)
         {
            // Forward sweeps, whose transpose is the backward sweep, so that
            // paired pre- and post-smoothing stays symmetric.
            AddLevel(system.Ptr(),
                     new GSSmoother(*system.As<SparseMatrix>(),
                                    GSSmoother::FORWARD, 1), false, true);
         }
#ifdef MFEM_USE_SUITESPARSE
         else
         {
            UMFPackSolver *coarse_solver = new UMFPackSolver;
            coarse_solver->SetOperator(*system);
            AddLevel(system.Ptr(), coarse_solver, false, true);
         }
#else
         else
         {
            CGSolver *coarse_solver = new CGSolver;
            coarse_solver->SetOperator(*system);
            coarse_solver->SetRelTol(1e-12);
            coarse_solver->SetAbsTol(0.0);
            coarse_solver->SetMaxIter(500);
            coarse_solver->SetPrintLevel(-1);
            AddLevel(system.Ptr(), coarse_solver, false, true);
         }
#endif
      }
   }
};

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-tri.mesh";
   const char *element = "argyris";
   int ref_levels = 2;
   bool w_cycle = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Affine triangular mesh.");
   args.AddOption(&element, "-e", "--element",
                  "C1 element to use: hct, rhct, bell, or argyris.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform multigrid refinements.");
   args.AddOption(&w_cycle, "-w", "--w-cycle", "-v", "--v-cycle",
                  "Use a W-cycle instead of a V-cycle.");
   args.ParseCheck();
   MFEM_VERIFY(ref_levels >= 0, "Refinement count must be nonnegative.");

   unique_ptr<FiniteElementCollection> fec = MakeCollection(element);
   Mesh *coarse_mesh = new Mesh(Mesh::LoadFromFile(mesh_file));
   MFEM_VERIFY(coarse_mesh->Dimension() == 2 &&
               coarse_mesh->SpaceDimension() == 2,
               "This miniapp requires a planar triangular mesh.");
   FiniteElementSpace *coarse_fes =
      new FiniteElementSpace(coarse_mesh, fec.get());
   FiniteElementSpaceHierarchy hierarchy(coarse_mesh, coarse_fes, true, true);
   for (int level = 0; level < ref_levels; level++)
   {
      hierarchy.AddUniformlyRefinedLevel(1, Ordering::byVDIM,
                                         Operator::MFEM_SPARSEMAT);
   }

   FiniteElementSpace &fine_fes = hierarchy.GetFinestFESpace();
   MFEM_VERIFY(fine_fes.GetMesh()->bdr_attributes.Size() > 0,
               "The clamped problem requires a mesh boundary.");
   Array<int> essential_boundary(fine_fes.GetMesh()->bdr_attributes.Max());
   essential_boundary = 1;

   ConstantCoefficient one(1.0);
   LinearForm b(&fine_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   GridFunction x(&fine_fes);
   x = 0.0;

   BiharmonicMultigrid multigrid(hierarchy, essential_boundary);
   multigrid.SetCycleType(w_cycle ? Multigrid::CycleType::WCYCLE :
                          Multigrid::CycleType::VCYCLE, 1, 1);

   OperatorPtr A;
   Vector B, X;
   multigrid.FormFineLinearSystem(x, b, A, X, B);

   cout << "\n         Geometric " << element << " hierarchy\n";
   cout << setw(8) << "Level" << setw(12) << "# Elems." << setw(12) << "# DOFs"
        << '\n';
   for (int i = 0; i < 32; i++) { cout << '='; }
   cout << '\n';
   for (int level = 0; level < hierarchy.GetNumLevels(); level++)
   {
      const FiniteElementSpace &fes = hierarchy.GetFESpaceAtLevel(level);
      cout << setw(8) << level << setw(12) << fes.GetMesh()->GetNE()
           << setw(12) << fes.GetTrueVSize() << '\n';
   }
   cout << '\n';

   CGSolver cg;
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(multigrid);
   cg.Mult(B, X);
   MFEM_VERIFY(cg.GetConverged(),
               "Biharmonic geometric multigrid did not converge.");
   multigrid.RecoverFineFEMSolution(X, b, x);

   return 0;
}

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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace
{

class DiagonalEAGradientIntegrator : public NonlinearFormIntegrator
{
public:
   real_t value = 1.0;

   void AssemblePA(const FiniteElementSpace &) override { }

   void AssembleGradEA(const Vector &, const FiniteElementSpace &fes,
                       Vector &ea_data) override
   {
      const int ne = fes.GetNE();
      const int elem_vdofs = fes.GetFE(0)->GetDof() * fes.GetVDim();
      auto A = Reshape(ea_data.HostReadWrite(), elem_vdofs, elem_vdofs, ne);
      for (int e = 0; e < ne; ++e)
      {
         for (int i = 0; i < elem_vdofs; ++i)
         {
            A(i, i, e) += value;
         }
      }
   }
};

} // namespace

TEST_CASE("NonlinearForm FULL gradient lifecycle",
          "[NonlinearForm][AssemblyLevel]")
{
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   CAPTURE(ordering);

   Mesh mesh = Mesh::MakeCartesian1D(2);
   H1_FECollection fec(1, 1);
   FiniteElementSpace fes(&mesh, &fec, 2, ordering);

   NonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(AssemblyLevel::FULL);
   auto *integ = new DiagonalEAGradientIntegrator;
   nlf.AddDomainIntegrator(integ);
   nlf.Setup();

   auto check_gradient = [&](const real_t value)
   {
      integ->value = value;
      Vector x(nlf.Width()), y(nlf.Height()), expected(nlf.Height());
      x = 1.0;
      expected = 0.0;

      Array<int> vdofs;
      for (int e = 0; e < fes.GetNE(); ++e)
      {
         fes.GetElementVDofs(e, vdofs);
         for (const int vdof : vdofs)
         {
            const int i = vdof >= 0 ? vdof : -1 - vdof;
            expected(i) += value;
         }
      }

      nlf.GetGradient(x).Mult(x, y);
      y -= expected;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
   };

   // Repeated gradients must not free the extension's reusable local matrix.
   check_gradient(1.0);
   check_gradient(2.0);

   // Updating the FE space must refresh the restriction and sparse graph.
   mesh.UniformRefinement();
   fes.Update();
   nlf.Update();
   nlf.Setup();
   check_gradient(3.0);
   check_gradient(4.0);
}

TEST_CASE("NonlinearForm Boundary Integrator", "[NonlinearForm]")
{
   // See problem description in ex27.

   Mesh mesh("./data/holes.mesh", 1, 1);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> nbc_bdr(mesh.bdr_attributes.Max());
   Array<int> rbc_bdr(mesh.bdr_attributes.Max());
   Array<int> dbc_bdr(mesh.bdr_attributes.Max());
   nbc_bdr = 0; nbc_bdr[0] = 1;
   rbc_bdr = 0; rbc_bdr[1] = 1;
   dbc_bdr = 0; dbc_bdr[2] = 1;

   Array<int> ess_tdof_list(0);
   fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);

   // See defaults in ex27.
   ConstantCoefficient matCoef(1.0);
   ConstantCoefficient dbcCoef(0.0);
   ConstantCoefficient nbcCoef(1.0);
   ConstantCoefficient rbcACoef(1.0);
   ConstantCoefficient rbcBCoef(1.0);
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   GridFunction u1(&fespace), u2(&fespace);
   u1 = 0.0;
   u2 = 0.0;
   u1.ProjectBdrCoefficient(dbcCoef, dbc_bdr);
   u2.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

   LinearForm b(&fespace);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   b.Assemble();

   // Solve as a linear problem.
   {
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, u1, b, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, u1);
   }

   // Solve as a nonlinear problem.
   {
      NonlinearForm a_nf(&fespace);
      a_nf.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a_nf.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a_nf.SetEssentialTrueDofs(ess_tdof_list);

      IterativeSolver::PrintLevel print;
      print.Iterations();
      CGSolver cg;
      cg.SetPrintLevel(print);
      cg.SetMaxIter(100);
      cg.SetRelTol(1e-12); cg.SetAbsTol(0.0);

      NewtonSolver newton;
      newton.iterative_mode = false;
      newton.SetSolver(cg);
      newton.SetOperator(a_nf);
      newton.SetPrintLevel(print);
      newton.SetRelTol(1e-14); newton.SetAbsTol(0.0);
      newton.SetMaxIter(1);

      newton.Mult(b, u2);
   }

   u2 -= u1;
   REQUIRE(u2.Norml2() == MFEM_Approx(0.0, 1e-5));
}

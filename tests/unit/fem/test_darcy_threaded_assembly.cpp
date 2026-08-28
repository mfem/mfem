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

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

using namespace mfem;

namespace darcy_threaded_assembly
{

// The mixed Darcy problem of test_darcy_hybridization.cpp, which is that of
// examples/hdg/ex5.cpp. Nothing here tests the discretisation -- what is under
// test is that DarcyHybridization::AssemblyMode::Threaded builds the identical
// trace matrix, so the problem only has to be one whose assembly exercises the
// blocks: A, the Schur complement, and the E, G and H face terms.

real_t pExact(const Vector &x)
{
   const real_t z = (x.Size() == 3) ? x(2) : 0.0;
   return exp(x(0)) * sin(x(1)) * cos(z);
}

real_t gExact(const Vector &x)
{
   return (x.Size() == 3) ? -pExact(x) : 0.0;
}

real_t pNatural(const Vector &x) { return -pExact(x); }

enum class Form { RT, DG };

/// A copy of the assembled trace matrix, in the only form that can settle the
/// question: the raw arrays, so the comparison is of stored entries and of the
/// pattern that holds them, not of an operator's action on some vector.
struct TraceMatrix
{
   int height = 0, width = 0, nnz = 0;
   Array<int> I, J;
   Vector A;

   void CopyFrom(const SparseMatrix &H)
   {
      height = H.Height();
      width  = H.Width();
      nnz    = H.NumNonZeroElems();
      I.SetSize(height+1);
      for (int i = 0; i <= height; i++) { I[i] = H.GetI()[i]; }
      J.SetSize(nnz);
      A.SetSize(nnz);
      for (int k = 0; k < nnz; k++) { J[k] = H.GetJ()[k]; A[k] = H.GetData()[k]; }
   }
};

/// Assemble the hybridized trace matrix once, in the given assembly mode.
TraceMatrix Assemble(Mesh &mesh, int order, Form form,
                     DarcyHybridization::AssemblyMode mode)
{
   const int dim = mesh.Dimension();

   std::unique_ptr<FiniteElementCollection> u_coll;
   if (form == Form::DG)
   {
      u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   }
   else
   {
      u_coll.reset(new RT_FECollection(order, dim));
   }
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, u_coll.get(), (form == Form::DG) ? dim : 1);
   FiniteElementSpace fes_p(&mesh, &p_coll);

   ConstantCoefficient k(1.0);
   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f)
   {
      f = 0.0;
   });
   FunctionCoefficient gcoeff(gExact);
   FunctionCoefficient natcoeff(pNatural);
   RatioCoefficient ik(1.0, k);

   DarcyForm darcy(&fes_u, &fes_p);
   LinearForm *fform = darcy.GetFluxRHS();

   if (form == Form::DG)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(k));
      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      B->AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.)));
      // The stabilisation is what puts the E and G face blocks into the Schur
      // complement, so the DG case covers strictly more of ComputeH() than RT.
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(ik, 0.5));

      fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(natcoeff));
   }
   else
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorFEMassIntegrator(k));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorFEDivergenceIntegrator);

      fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
      fform->AddBoundaryIntegrator(
         new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(&mesh, &trace_coll);

   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs);
   darcy.GetHybridization()->SetAssemblyMode(mode);
   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, B, true);

   SparseMatrix *H = A.As<SparseMatrix>();
   MFEM_VERIFY(H, "the hybridized system is not an assembled SparseMatrix");

   TraceMatrix out;
   out.CopyFrom(*H);
   return out;
}

/** The comparison the acceptance criterion asks for, and it is equality
    rather than a tolerance on purpose.

    The plan this came from warned that "bitwise agreement is the wrong
    criterion and asking for it will send someone chasing a non-existent bug",
    on the grounds that threading a reduction reassociates it. That is true of
    a threaded scatter and false here, because the element-local work is
    per-element and reassociates nothing: every entry of the trace matrix is a
    sum of at most two terms, one per element sharing the face the trace dof
    lives on, and IEEE addition of two terms is order-independent. Equality is
    therefore the right test and a difference of any size is a defect.

    This is a *differential* test: it compares the two modes, which share the
    scatter, so it cannot catch a change that moves both. What guards that is
    the rest of the suite together with the miniapp regressions. What it does
    catch is the hazard the plan named -- reinstating one `static` on the
    element-local scratch, so two threads share it, fails this at four threads
    on the smallest case here, with 143 stored entries against 144. */
void RequireIdentical(const TraceMatrix &ref, const TraceMatrix &got)
{
   REQUIRE(got.height == ref.height);
   REQUIRE(got.width == ref.width);
   // The pattern first: AddSubMatrix()'s skip_zeros drops entries by value, so
   // a scatter that had lost the &rows != &cols distinction would show up here
   // as a different nnz long before any entry differed.
   REQUIRE(got.nnz == ref.nnz);
   for (int i = 0; i <= ref.height; i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int k = 0; k < ref.nnz; k++) { REQUIRE(got.J[k] == ref.J[k]); }

   real_t max_diff = 0.0;
   for (int k = 0; k < ref.nnz; k++)
   {
      max_diff = std::max(max_diff, std::abs(got.A[k] - ref.A[k]));
   }
   REQUIRE(max_diff == 0.0);
}

} // namespace darcy_threaded_assembly

using namespace darcy_threaded_assembly;

TEST_CASE("Threaded trace assembly is bit-for-bit the serial one",
          "[DarcyHybridization][AssemblyMode]")
{
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   const int saved_threads = omp_get_max_threads();

   for (Form form : {Form::RT, Form::DG})
   {
      for (int order : {0, 1, 2})
      {
         // Two dimensions and two element types, because the chunking walks
         // elements in blocks and a ragged block size is what would break it.
         Mesh quad = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
         Mesh hex  = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

         for (Mesh *mesh : {&quad, &hex})
         {
            CAPTURE(int(form), order, mesh->Dimension());

            omp_set_num_threads(1);
            const TraceMatrix ref = Assemble(
                                       *mesh, order, form,
                                       DarcyHybridization::AssemblyMode::Serial);

            // A thread count of one must also agree, which separates "the
            // refactor changed something" from "the threading changed
            // something": only the second can depend on the count.
            for (int nt : {1, 2, 4, 8})
            {
               CAPTURE(nt);
               omp_set_num_threads(nt);
               const TraceMatrix got = Assemble(
                                          *mesh, order, form,
                                          DarcyHybridization::AssemblyMode::Threaded);
               RequireIdentical(ref, got);
            }
         }
      }
   }

   omp_set_num_threads(saved_threads);
#else
   WARN("Threaded assembly needs MFEM_USE_OPENMP and MFEM_THREAD_SAFE; "
        "this build has neither or only one, so nothing was checked.");
#endif
}

TEST_CASE("Serial assembly mode is the default",
          "[DarcyHybridization][AssemblyMode]")
{
   // The acceptance criterion "a serial build unchanged" rests on this: every
   // existing caller gets the loop it always had without saying anything.
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   L2_FECollection u_coll(0, 2, BasisType::GaussLobatto);
   L2_FECollection p_coll(0, 2);
   FiniteElementSpace fes_u(&mesh, &u_coll, 2);
   FiniteElementSpace fes_p(&mesh, &p_coll);
   DarcyForm darcy(&fes_u, &fes_p);

   Array<int> ess;
   DG_Interface_FECollection trace_coll(0, 2);
   FiniteElementSpace fes_t(&mesh, &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(), ess);

   REQUIRE(darcy.GetHybridization()->GetAssemblyMode() ==
           DarcyHybridization::AssemblyMode::Serial);
}

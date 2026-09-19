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

#include <vector>

using namespace mfem;

namespace integ_thread_safe
{

/** @brief Assemble every element matrix of @a mesh with ONE integrator object,
    optionally from several threads at once, and return them.

    The integrator is deliberately shared: that is the whole point. MFEM's
    convention is that an integrator's per-point scratch lives behind
    `#ifndef MFEM_THREAD_SAFE`, so that a build which sets that option has no
    shared members left and the same object may be entered concurrently. A
    class that declares its scratch outside the guard silently ignores the
    option, and the failure is a wrong element matrix rather than a crash. */
template <typename Integ>
std::vector<DenseMatrix> AssembleAll(Mesh &mesh, FiniteElementSpace &fes,
                                     Integ &integ, int nthreads)
{
   const int NE = mesh.GetNE();
   std::vector<DenseMatrix> out(NE);

#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
   {
      // Caller-supplied, because Mesh keeps ONE Transformation for the whole
      // mesh and handing out a pointer to it is its own race -- a different
      // one from the integrator's, and not what this case is about.
      IsoparametricTransformation Tr;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int e = 0; e < NE; e++)
      {
         mesh.GetElementTransformation(e, &Tr);
         integ.AssembleElementMatrix(*fes.GetFE(e), Tr, out[e]);
      }
   }
   return out;
}

/// The mixed shape: trial and test spaces differ, so AssembleElementMatrix2().
template <typename Integ>
std::vector<DenseMatrix> AssembleAll2(Mesh &mesh, FiniteElementSpace &trial,
                                      FiniteElementSpace &test, Integ &integ,
                                      int nthreads)
{
   const int NE = mesh.GetNE();
   std::vector<DenseMatrix> out(NE);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
   {
      IsoparametricTransformation Tr;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int e = 0; e < NE; e++)
      {
         mesh.GetElementTransformation(e, &Tr);
         integ.AssembleElementMatrix2(*trial.GetFE(e), *test.GetFE(e), Tr,
                                      out[e]);
      }
   }
   return out;
}

/// The linear-form shape: AssembleRHSElementVect(), over elements or over
/// boundary elements.
template <typename Integ>
std::vector<Vector> AssembleAllRHS(Mesh &mesh, FiniteElementSpace &fes,
                                   Integ &integ, int nthreads, bool bdr)
{
   const int N = bdr ? mesh.GetNBE() : mesh.GetNE();
   std::vector<Vector> out(N);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
   {
      IsoparametricTransformation Tr;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int e = 0; e < N; e++)
      {
         if (bdr)
         {
            mesh.GetBdrElementTransformation(e, &Tr);
            integ.AssembleRHSElementVect(*fes.GetBE(e), Tr, out[e]);
         }
         else
         {
            mesh.GetElementTransformation(e, &Tr);
            integ.AssembleRHSElementVect(*fes.GetFE(e), Tr, out[e]);
         }
      }
   }
   return out;
}

/// Bitwise, for the reason the first case gives: no sum crosses a thread.
template <typename T>
void RequireIdenticalAll(const std::vector<T> &got, const std::vector<T> &ref)
{
   REQUIRE(got.size() == ref.size());
   real_t max_diff = 0.0;
   for (size_t e = 0; e < ref.size(); e++)
   {
      REQUIRE(got[e].GetData() != NULL);
      REQUIRE(got[e].Size() == ref[e].Size());
      for (int i = 0; i < ref[e].Size(); i++)
      {
         max_diff = std::max(max_diff,
                             std::abs(got[e].GetData()[i] - ref[e].GetData()[i]));
      }
   }
   REQUIRE(max_diff == 0.0);
}

} // namespace integ_thread_safe

using namespace integ_thread_safe;

TEST_CASE("VectorMassIntegrator assembles the same matrices on many threads",
          "[BilinearFormIntegrator][ThreadSafe]")
{
   // **VectorMassIntegrator declared its scratch OUTSIDE
   // `#ifndef MFEM_THREAD_SAFE`** -- `shape`, `te_shape`, `vec`, `partelmat`
   // and `mcoeff` were plain private members -- so a MFEM_THREAD_SAFE build
   // got no thread safety from the option it had set, and two threads
   // assembling different elements wrote each other's `partelmat` mid-sum.
   //
   // The symptom is not a crash. It is an element matrix that is right on one
   // thread and quietly wrong on several: measured downstream, a hybridized
   // HDG Newton whose flux block came back as the same constant-coefficient
   // mass matrix for every element at one thread and differently wrong for a
   // handful of them at four, surfacing only as a NaN once the solver had
   // used the bad Jacobian. So the assertion here is on the MATRICES, not on
   // whether anything aborted.
   //
   // Equality is BITWISE and that is a claim about the arithmetic rather than
   // optimism: each element's matrix is computed by one thread from its own
   // quadrature loop, and no sum crosses a thread, so a correct
   // implementation must reproduce the serial answer exactly. Anything less
   // than bitwise would mean threads are sharing something.
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   H1_FECollection fec(2, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   const int saved = omp_get_max_threads();

   // Three coefficient shapes, because they take three different branches
   // through the routine and only one of them touches `mcoeff`.
   const int which = GENERATE(0, 1, 2);
   CAPTURE(which);

   ConstantCoefficient q(2.5);
   Vector vq_v(dim);
   for (int d = 0; d < dim; d++) { vq_v(d) = 1.0 + d; }
   VectorConstantCoefficient vq(vq_v);
   DenseMatrix mq_m(dim);
   mq_m = 0.0;
   for (int d = 0; d < dim; d++) { mq_m(d, d) = 1.0 + 0.5*d; }
   MatrixConstantCoefficient mq(mq_m);

   auto make = [&]() -> VectorMassIntegrator *
   {
      if (which == 0) { return new VectorMassIntegrator(q); }
      if (which == 1) { return new VectorMassIntegrator(vq); }
      return new VectorMassIntegrator(mq);
   };

   std::unique_ptr<VectorMassIntegrator> serial_integ(make());
   const std::vector<DenseMatrix> ref =
      AssembleAll(mesh, fes, *serial_integ, 1);

   for (int nt : {2, 4, 8})
   {
      CAPTURE(nt);
      omp_set_num_threads(nt);

      // A FRESH integrator per arm, so the comparison is against a clean
      // object rather than one the serial pass has already sized.
      std::unique_ptr<VectorMassIntegrator> integ(make());
      const std::vector<DenseMatrix> got =
         AssembleAll(mesh, fes, *integ, nt);

      REQUIRE(got.size() == ref.size());
      real_t max_diff = 0.0;
      for (size_t e = 0; e < ref.size(); e++)
      {
         REQUIRE(got[e].Height() == ref[e].Height());
         REQUIRE(got[e].Width() == ref[e].Width());
         for (int i = 0; i < ref[e].Height(); i++)
         {
            for (int j = 0; j < ref[e].Width(); j++)
            {
               max_diff = std::max(max_diff,
                                   std::abs(got[e](i, j) - ref[e](i, j)));
            }
         }
      }
      REQUIRE(max_diff == 0.0);
   }

   omp_set_num_threads(saved);
#else
   WARN("Integrator thread safety needs MFEM_USE_OPENMP and MFEM_THREAD_SAFE; "
        "this build has neither or only one, so nothing was checked. That is "
        "precisely how the defect this case pins survived: the option that is "
        "supposed to deliver the guarantee is off in almost every build, so "
        "the classes that ignore it are never contradicted.");
#endif
}

TEST_CASE("The integrators meq and gffp install are reentrant",
          "[BilinearFormIntegrator][LinearFormIntegrator][ThreadSafe]")
{
   // The set is bounded by USE rather than by sweeping the library: these are
   // the MFEM stock integrators that /home/ian/projects/meq and
   // /home/ian/projects/gffp actually construct, read out of their sources.
   // Of those, MassIntegrator and ConvectionIntegrator already guarded their
   // scratch and SumIntegrator has none, so what is left is what this case
   // covers. VectorMassIntegrator has a case of its own above.
   //
   // Two of these are not incidental. An HDG assembly leg was measured at
   // 9.68% of a run with 92% of it inside two element loops, and the only two
   // unguarded integrators those loops reach are VectorMassIntegrator and
   // VectorDivergenceIntegrator -- so these two are exactly what stands
   // between that leg and being threaded at all.
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   const int dim = 2;
   H1_FECollection fec(2, dim), sfec(1, dim);
   FiniteElementSpace vfes(&mesh, &fec, dim), sfes(&mesh, &sfec);
   ConstantCoefficient q(2.5);
   Vector vv(dim); vv = 1.5;
   VectorConstantCoefficient vq(vv);

   const int saved = omp_get_max_threads();

   SECTION("DiffusionIntegrator, whose four scratch Vectors sat OUTSIDE a "
           "guard the same class already had for everything else")
   {
      DiffusionIntegrator ref_i(q);
      const auto ref = AssembleAll(mesh, sfes, ref_i, 1);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         DiffusionIntegrator got_i(q);
         RequireIdenticalAll(AssembleAll(mesh, sfes, got_i, nt), ref);
      }
   }

   SECTION("VectorDivergenceIntegrator, the largest single piece of the "
           "hybridized assembly leg")
   {
      VectorDivergenceIntegrator ref_i(q);
      const auto ref = AssembleAll2(mesh, vfes, sfes, ref_i, 1);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         VectorDivergenceIntegrator got_i(q);
         RequireIdenticalAll(AssembleAll2(mesh, vfes, sfes, got_i, nt), ref);
      }
   }

   SECTION("TransposeIntegrator, whose own bfi_elmat was unguarded even when "
           "the integrator it wraps was not")
   {
      // Wrapping a guarded integrator isolates the WRAPPER: any difference
      // here is bfi_elmat and nothing else.
      TransposeIntegrator ref_i(new MassIntegrator(q));
      const auto ref = AssembleAll(mesh, sfes, ref_i, 1);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         TransposeIntegrator got_i(new MassIntegrator(q));
         RequireIdenticalAll(AssembleAll(mesh, sfes, got_i, nt), ref);
      }
   }

   SECTION("DomainLFIntegrator and VectorDomainLFIntegrator")
   {
      DomainLFIntegrator ref_s(q);
      const auto ref_sv = AssembleAllRHS(mesh, sfes, ref_s, 1, false);
      VectorDomainLFIntegrator ref_v(vq);
      const auto ref_vv = AssembleAllRHS(mesh, vfes, ref_v, 1, false);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         DomainLFIntegrator got_s(q);
         RequireIdenticalAll(AssembleAllRHS(mesh, sfes, got_s, nt, false),
                             ref_sv);
         VectorDomainLFIntegrator got_v(vq);
         RequireIdenticalAll(AssembleAllRHS(mesh, vfes, got_v, nt, false),
                             ref_vv);
      }
   }

   SECTION("VectorBoundaryFluxLFIntegrator, over boundary elements")
   {
      VectorBoundaryFluxLFIntegrator ref_i(q);
      const auto ref = AssembleAllRHS(mesh, sfes, ref_i, 1, true);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         VectorBoundaryFluxLFIntegrator got_i(q);
         RequireIdenticalAll(AssembleAllRHS(mesh, sfes, got_i, nt, true), ref);
      }
   }

   omp_set_num_threads(saved);
#else
   WARN("Integrator thread safety needs MFEM_USE_OPENMP and MFEM_THREAD_SAFE.");
#endif
}

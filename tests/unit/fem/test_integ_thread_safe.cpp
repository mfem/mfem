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

/** @brief The same, for element matrices, and it is NOT redundant with the
    template above.

    `DenseMatrix::Size()` is `Width()` -- "for backward compatibility define
    Size to be synonym of Width", densemat.hpp -- so the generic routine walks
    `GetData()[0 .. Width())`, which in MFEM's column-major layout is the
    FIRST COLUMN and nothing else. On a 12x12 block that is 12 of 144 entries,
    and a difference confined to any other column passes it. Nor does it
    compare the heights. This overload is chosen ahead of the template by
    exact match, so the call sites above get it with no change to them. */
void RequireIdenticalAll(const std::vector<DenseMatrix> &got,
                         const std::vector<DenseMatrix> &ref)
{
   REQUIRE(got.size() == ref.size());
   real_t max_diff = 0.0;
   for (size_t e = 0; e < ref.size(); e++)
   {
      REQUIRE(got[e].Height() == ref[e].Height());
      REQUIRE(got[e].Width() == ref[e].Width());
      for (int j = 0; j < ref[e].Width(); j++)
         for (int i = 0; i < ref[e].Height(); i++)
         {
            max_diff = std::max(max_diff, std::abs(got[e](i,j) - ref[e](i,j)));
         }
   }
   REQUIRE(max_diff == 0.0);
}

/** @brief Assemble the boundary-face matrix of every face carrying
    @a bdr_attr with ONE integrator object, optionally from several threads at
    once, and return them.

    The face counterpart of AssembleAll(). Two things are resolved SERIALLY
    before the parallel region and that is deliberate in both cases. The face
    transformations are taken in the user-allocated form, because
    Mesh::GetBdrFaceTransformations()'s pointer form hands out the Mesh's own
    single object and sharing that is a race of the Mesh's, not of the
    integrator's. And the finite elements are looked up in a warm-up pass, so
    that a lazily-built lookup inside FiniteElementSpace or the geometry
    tables cannot be what a later difference is measuring. What is left
    shared across the threads is the integrator and nothing else. */
template <typename Integ>
std::vector<DenseMatrix> AssembleAllBdrFace(Mesh &mesh,
                                            FiniteElementSpace &fes,
                                            int bdr_attr, Integ &integ,
                                            int nthreads)
{
   std::vector<int> bes;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      if (mesh.GetBdrAttribute(be) == bdr_attr) { bes.push_back(be); }
   }
   const int N = static_cast<int>(bes.size());

   std::vector<const FiniteElement *> fe(N);
   {
      FaceElementTransformations FTr;
      IsoparametricTransformation T1, T2;
      for (int i = 0; i < N; i++)
      {
         mesh.GetBdrFaceTransformations(bes[i], FTr, T1, T2);
         fe[i] = fes.GetFE(FTr.Elem1No);
      }
   }

   std::vector<DenseMatrix> out(N);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
   {
      FaceElementTransformations FTr;
      IsoparametricTransformation T1, T2;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int i = 0; i < N; i++)
      {
         mesh.GetBdrFaceTransformations(bes[i], FTr, T1, T2);
         // Elem2 is negative on a boundary face and the second element is
         // never read, so el1 is passed twice.
         integ.AssembleFaceMatrix(*fe[i], *fe[i], FTr, out[i]);
      }
   }
   return out;
}

/** @brief HDGExtensionIntegrator::ComputeLift() at every quadrature point of
    every face of @a bdr_attr, one Vector per face.

    A separate public entry point with its own use of the same scratch, and
    the one PathLiftCoefficient -- and through it the eta_5 face indicator --
    actually reaches. Same serial resolution of the geometry and the dofs as
    AssembleAllBdrFace(), for the same reason. */
std::vector<Vector> LiftAllBdrFace(Mesh &mesh, FiniteElementSpace &fes,
                                   int bdr_attr, const GridFunction &u,
                                   HDGExtensionIntegrator &integ,
                                   int ir_order, int nthreads)
{
   std::vector<int> bes;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      if (mesh.GetBdrAttribute(be) == bdr_attr) { bes.push_back(be); }
   }
   const int N = static_cast<int>(bes.size());

   std::vector<const FiniteElement *> fe(N);
   std::vector<Vector> elfun(N);
   {
      FaceElementTransformations FTr;
      IsoparametricTransformation T1, T2;
      Array<int> vdofs;
      for (int i = 0; i < N; i++)
      {
         mesh.GetBdrFaceTransformations(bes[i], FTr, T1, T2);
         fe[i] = fes.GetFE(FTr.Elem1No);
         fes.GetElementVDofs(FTr.Elem1No, vdofs);
         u.GetSubVector(vdofs, elfun[i]);
      }
   }

   std::vector<Vector> out(N);
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
   {
      FaceElementTransformations FTr;
      IsoparametricTransformation T1, T2;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int i = 0; i < N; i++)
      {
         mesh.GetBdrFaceTransformations(bes[i], FTr, T1, T2);
         const IntegrationRule &fir =
            IntRules.Get(FTr.GetGeometryType(), ir_order);
         out[i].SetSize(fir.GetNPoints());
         for (int q = 0; q < fir.GetNPoints(); q++)
         {
            out[i](q) = integ.ComputeLift(*fe[i], FTr, fir.IntPoint(q),
                                          elfun[i]);
         }
      }
   }
   return out;
}

/// Nothing may be compared to itself: a set of blocks that is all zero, or a
/// lifting that is identically nothing, would satisfy any equality assertion
/// whatever the integrator did.
template <typename T>
void RequireNotAllZero(const std::vector<T> &v)
{
   REQUIRE(v.size() > 0u);
   real_t biggest = 0.0;
   for (size_t e = 0; e < v.size(); e++)
   {
      for (int i = 0; i < v[e].Size(); i++)
      {
         biggest = std::max(biggest, std::abs(v[e].GetData()[i]));
      }
   }
   REQUIRE(biggest > 1e-8);
}

/// Omega is the disc of radius 0.45 about the centre of the unit square, so
/// that Gamma is analytic and ClosestPointPath -- which is stateless, and
/// therefore cannot be what a threaded difference is measuring -- supplies
/// the paths.
inline real_t ExtDiscR() { return 0.45; }

inline void ExtDiscCentre(Vector &c) { c.SetSize(2); c = 0.5; }

inline real_t ExtDiscPhi(const Vector &x)
{
   Vector c; ExtDiscCentre(c);
   return std::sqrt((x(0)-c(0))*(x(0)-c(0)) + (x(1)-c(1))*(x(1)-c(1)))
          - ExtDiscR();
}

/// D_h: the elements of an n x n triangulation of the unit square lying
/// wholly inside the disc, extracted as a SubMesh whose one boundary
/// attribute is Gamma_h.
struct ExtSubdomain
{
   std::unique_ptr<Mesh> background;
   std::unique_ptr<SubMesh> D_h;
   int gamma_h_attr{};
};

inline ExtSubdomain BuildExtSubdomain(int n)
{
   ExtSubdomain s;
   s.background = std::make_unique<Mesh>(
                     Mesh::MakeCartesian2D(n, n, Element::TRIANGLE));

   Array<int> marker;
   const int count = MarkLevelSetSubdomain(*s.background, ExtDiscPhi, 0.,
                                           marker);
   REQUIRE(count > 0);
   for (int i = 0; i < s.background->GetNE(); i++)
   {
      s.background->SetAttribute(i, marker[i] ? 1 : 2);
   }
   s.background->SetAttributes();

   Array<int> domain_attr(1);
   domain_attr[0] = 1;
   s.D_h = std::make_unique<SubMesh>(
              SubMesh::CreateFromDomain(*s.background, domain_attr));
   REQUIRE(s.D_h->bdr_attributes.Size() == 1);
   s.gamma_h_attr = s.D_h->bdr_attributes.Max();
   return s;
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


TEST_CASE("HDGExtensionIntegrator assembles the same face blocks on many "
          "threads", "[BilinearFormIntegrator][ThreadSafe][Extension]")
{
   // **HDGExtensionIntegrator declared its whole working set as plain
   // members** -- eight Vectors, two DenseMatrices and, worst of the group,
   // an ElementExtension. So a MFEM_THREAD_SAFE build got nothing from the
   // option it had set.
   //
   // The ElementExtension is the one that makes this more than a scaled-down
   // copy of the VectorMassIntegrator case above. It holds an
   // InverseElementTransformation that AssembleFaceMatrix() points at the
   // element owning the face, and LiftBasis() then inverts that map once per
   // point of the path, per basis function. Two threads on two faces of two
   // different elements therefore do not merely overwrite each other's
   // arithmetic: the later SetElement() wins and the earlier thread inverts
   // the WRONG ELEMENT'S map for the rest of its face, so its extension is
   // some other element's polynomial. There is no abort in that; the reference
   // point comes back, the shape functions evaluate, and the block is wrong.
   //
   // Why this integrator rather than any other in fem/darcy: meq constructs
   // it more often than any MFEM stock integrator, it sits on the flux mass
   // form, and the flux mass form is assembled inside the element loops the
   // sibling branch threads. The guard is therefore not reachable from this
   // branch at all -- there is no `#pragma omp parallel` anywhere under
   // fem/darcy here -- and is reachable on the tree that merges both. That is
   // also why it has to be tested here, against the integrator directly:
   // waiting for a threaded caller to exist on this branch would mean never
   // testing it.
   //
   // Equality is BITWISE, on the same argument the first case gives: each
   // face's block is computed by one thread from its own quadrature loop and
   // no sum crosses a thread. The only arithmetic here that could reassociate
   // under a thread count is the small dense solve inside
   // InverseElementTransformation, at 2x2; MKL_NUM_THREADS is pinned for the
   // suite in any case, and the constant coefficients below are chosen partly
   // so that nothing else with state of its own is in the loop -- a
   // FunctionCoefficient carries an unguarded `mutable Vector transip` and
   // would be measuring MFEM's coefficient convention rather than this class.
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   const ExtSubdomain s = BuildExtSubdomain(16);
   Mesh &D_h = *s.D_h;

   const int dim = 2;
   const int order = 2;
   L2_FECollection fec(order, dim);
   FiniteElementSpace fes(&D_h, &fec, dim, Ordering::byNODES);

   Vector c; ExtDiscCentre(c);
   const ClosestPointPath path(ClosestPointPath::Sphere(c, ExtDiscR()));

   ConstantCoefficient C(1.5);
   DenseMatrix Cm(dim);
   Cm(0,0) = 1.5; Cm(0,1) = 0.25; Cm(1,0) = -0.25; Cm(1,1) = 0.75;
   MatrixConstantCoefficient MC(Cm);

   const int saved = omp_get_max_threads();

   SECTION("the assembled block, with a scalar inverse diffusion")
   {
      HDGExtensionIntegrator ref_i(path, C);
      const auto ref = AssembleAllBdrFace(D_h, fes, s.gamma_h_attr, ref_i, 1);
      RequireNotAllZero(ref);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         // A FRESH integrator per arm, so the comparison is against a clean
         // object and not one the serial pass has already sized.
         HDGExtensionIntegrator got_i(path, C);
         RequireIdenticalAll(
            AssembleAllBdrFace(D_h, fes, s.gamma_h_attr, got_i, nt), ref);
      }
   }

   SECTION("the assembled block, with a matrix inverse diffusion")
   {
      // The MatrixCoefficient branch is the only one that reaches Cmat, and
      // it takes a different path through LiftBasis(); a guard that missed
      // one member would show up here and not above. The matrix is
      // deliberately NOT symmetric, so a transpose that went astray between
      // threads cannot cancel.
      HDGExtensionIntegrator ref_i(path, MC);
      const auto ref = AssembleAllBdrFace(D_h, fes, s.gamma_h_attr, ref_i, 1);
      RequireNotAllZero(ref);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         HDGExtensionIntegrator got_i(path, MC);
         RequireIdenticalAll(
            AssembleAllBdrFace(D_h, fes, s.gamma_h_attr, got_i, nt), ref);
      }
   }

   SECTION("ComputeLift(), which PathLiftCoefficient drives")
   {
      // ComputeLift() uses the same members from a different entry point,
      // and it is what PathLiftCoefficient and the eta_5 indicator call.
      GridFunction u(&fes);
      u.Randomize(1234);

      HDGExtensionIntegrator ref_i(path, C);
      const auto ref =
         LiftAllBdrFace(D_h, fes, s.gamma_h_attr, u, ref_i, 2*order + 2, 1);
      RequireNotAllZero(ref);
      for (int nt : {2, 4, 8})
      {
         CAPTURE(nt);
         omp_set_num_threads(nt);
         HDGExtensionIntegrator got_i(path, C);
         RequireIdenticalAll(
            LiftAllBdrFace(D_h, fes, s.gamma_h_attr, u, got_i, 2*order + 2, nt),
            ref);
      }
   }

   omp_set_num_threads(saved);
#else
   WARN("HDGExtensionIntegrator thread safety needs MFEM_USE_OPENMP and "
        "MFEM_THREAD_SAFE; this build has neither or only one, so nothing "
        "was checked. Both HDG development trees have them off, which is "
        "exactly how a class that ignores the option goes uncontradicted.");
#endif
}

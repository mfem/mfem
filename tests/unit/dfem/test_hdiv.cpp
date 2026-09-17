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


// Mass and DivDiv integrators for H(div) spaces in dFEM

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

// Note: the transformation from reference to physical for Hdiv is
//       w = J w_ref / det(J)
//       div w = div_ref w_ref / det(J)

// ────────────────────────────────────────────────────────────────────────────
// (u, v) on an H(div) space, in reference coordinates.
// (u, v)         = \int (1/det J) (J^T J u_ref) . v_ref  dxi
template <int DIM> struct hdiv_mass_qf
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<dscalar_t, DIM> &u,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      tensor<dscalar_t, DIM> &v) const
   {
      v = (w / det(J)) * dot(transpose(J), dot(J, u));
   }
};

// ────────────────────────────────────────────────────────────────────────────
// (div u, div v) on an H(div) space, in reference coordinates.
// (div u, div v) = \int (1/det J) div_ref u_ref div_ref v_ref  dxi
template <int DIM> struct hdiv_divdiv_qf
{
   MFEM_HOST_DEVICE inline void operator()(
      const dscalar_t &du,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      dscalar_t &dv) const
   {
      dv = du * (w / det(J));
   }
};

// ────────────────────────────────────────────────────────────────────────────
// (u, v) + (div u, div v) on an H(div) space, in reference coordinates.
//
// Smoke test for Hdiv multiple outputs
template <int DIM> struct hdiv_mass_divdiv_qf
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<dscalar_t, DIM> &u,
      const dscalar_t &du,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      tensor<dscalar_t, DIM> &v,
      dscalar_t &dv) const
   {
      const real_t c = w / det(J);
      v = c * dot(transpose(J), dot(J, u));
      dv = du * c;
   }
};

// TODO: to add full RT-L2 support we might want to modify restriction.cpp
// to allow FillSparseMatrix to accept L2ElementRestriction as well.


// (div u, p), with an H(div) trial field and an H1 test field.
struct hdiv_to_h1_qf
{
   MFEM_HOST_DEVICE inline void operator()(const dscalar_t &du,
                                           const real_t &w,
                                           dscalar_t &p) const
   {
      p = w * du;
   }
};

// (p, div u), the transpose block with an H1 trial field and H(div) test field.
struct h1_to_hdiv_qf
{
   MFEM_HOST_DEVICE inline void operator()(const dscalar_t &p,
                                           const real_t &w,
                                           dscalar_t &du) const
   {
      du = w * p;
   }
};

// ────────────────────────────────────────────────────────────────────────────
struct HdivSetup
{
   ParMesh pmesh;
   ParGridFunction *nodes = nullptr;
   RT_FECollection fec;
   ParFiniteElementSpace pfes;
   const IntegrationRule *ir = nullptr;
   Array<int> all_domain_attr;
   Vector N;

   HdivSetup(const char *filename, int dim, int p)
      : HdivSetup(Mesh(filename), dim, p) {}

   // ParMesh takes an lvalue, so the serial mesh is named here and dropped
   // once this constructor returns.
   HdivSetup(Mesh &&smesh, int dim, int p)
      : pmesh(MPI_COMM_WORLD, smesh), fec(p, dim), pfes(&pmesh, &fec)
   {
      pmesh.EnsureNodes();
      nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
      // The RT closed basis has p + 2 dofs per direction; integrate the mass
      // and div-div forms of that basis exactly on an affine element.
      ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * (p + 2) + 2);
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }
      nodes->GetTrueDofs(N);
   }
};

real_t HdivMaxError(MPI_Comm comm, const Vector &a, const Vector &b)
{
   Vector d(a);
   d -= b;
   const real_t local = d.Normlinf();
   real_t global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
}

// ────────────────────────────────────────────────────────────────────────────
/// Which bilinear form MFEM builds as the reference.
enum class HdivForm { Mass, DivDiv, MassDivDiv };

void AddHdivIntegrators(ParBilinearForm &blf, HdivForm form,
                        const IntegrationRule *ir)
{
   auto add = [&](BilinearFormIntegrator *bfi)
   {
      bfi->SetIntRule(ir);
      blf.AddDomainIntegrator(bfi);
   };

   switch (form)
   {
      case HdivForm::Mass:
         add(new VectorFEMassIntegrator());
         break;
      case HdivForm::DivDiv:
         add(new DivDivIntegrator());
         break;
      case HdivForm::MassDivDiv:
         add(new VectorFEMassIntegrator());
         add(new DivDivIntegrator());
         break;
   }
}

// ────────────────────────────────────────────────────────────────────────────
template <int DIM, typename inputs_t, typename outputs_t, typename qf_t>
void CheckHdivOperator(HdivSetup &setup, qf_t qf, HdivForm form)
{
   ParFiniteElementSpace &pfes = setup.pfes;
   const int tvsize = pfes.GetTrueVSize();
   const MPI_Comm comm = setup.pmesh.GetComm();

   static constexpr int U = 0, Coords = 1;
   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, setup.nodes->ParFESpace() }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

   ParGridFunction x(&pfes), y(&pfes);
   Vector X(tvsize), Y(tvsize), Z(tvsize);
   X.Randomize(1);
   x.SetFromTrueDofs(X);

   // Reference: the same bilinear form assembled by MFEM.
   ParBilinearForm blf_pa(&pfes);
   AddHdivIntegrators(blf_pa, form, setup.ir);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.Assemble();

   blf_pa.Mult(x, y);
   pfes.GetProlongationMatrix()->MultTranspose(y, Y);
   REQUIRE(Y.Normlinf() > 1e-8); // guard against comparing zeros

   SECTION("Action")
   {
      DifferentiableOperator dop(in_fds, out_fds, setup.pmesh);
      dop.AddDomainIntegrator<LocalQFBackend>(
         qf, inputs_t {}, outputs_t {}, *setup.ir, setup.all_domain_attr);

      MultiVector MX{ X, setup.N }, MZ{ Z };
      dop.Mult(MX, MZ);
      REQUIRE(HdivMaxError(comm, Y, Z) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("Derivative action, MF")
   {
      DifferentiableOperator dop(in_fds, out_fds, setup.pmesh);
      constexpr auto kernels = DerivativeKernels::Action;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs_t {}, outputs_t {}, *setup.ir, setup.all_domain_attr,
         Derivatives<U> {});

      // Differentiate at X on randomized direction dX.
      ParGridFunction dx(&pfes), dy(&pfes);
      Vector dX(tvsize), dY(tvsize), dZ(tvsize);
      dX.Randomize(2);
      dx.SetFromTrueDofs(dX);
      blf_pa.Mult(dx, dy);
      pfes.GetProlongationMatrix()->MultTranspose(dy, dY);
      REQUIRE(dY.Normlinf() > 1e-8);

      MultiVector MX{ X, setup.N }, MdZ{ dZ };

      // Both forms are linear in U, so the derivative action along dX is the
      // reference operator applied to dX.
      auto dRdU = dop.GetDerivative(U, MX, false);
      dRdU->Mult(dX, MdZ);
      REQUIRE(HdivMaxError(comm, dY, dZ) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("Derivative action, cached")
   {
      DifferentiableOperator dop(in_fds, out_fds, setup.pmesh);
      constexpr auto kernels =
         DerivativeKernels::Action | DerivativeKernels::Apply;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs_t {}, outputs_t {}, *setup.ir, setup.all_domain_attr,
         Derivatives<U> {});

      // Differentiate at X on randomized direction dX.
      ParGridFunction dx(&pfes), dy(&pfes);
      Vector dX(tvsize), dY(tvsize), dZ(tvsize);
      dX.Randomize(2);
      dx.SetFromTrueDofs(dX);
      blf_pa.Mult(dx, dy);
      pfes.GetProlongationMatrix()->MultTranspose(dy, dY);
      REQUIRE(dY.Normlinf() > 1e-8);

      MultiVector MX{ X, setup.N }, MdZ{ dZ };

      // Both forms are linear in U, so the derivative action along dX is the
      // reference operator applied to dX.
      auto dRdU = dop.GetDerivative(U, MX, true);
      dRdU->Mult(dX, MdZ);
      REQUIRE(HdivMaxError(comm, dY, dZ) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("Assemble diagonal")
   {
      DifferentiableOperator dop(in_fds, out_fds, setup.pmesh);
      constexpr auto kernels = DerivativeKernels::AssembleDiagonal;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs_t {}, outputs_t {}, *setup.ir, setup.all_domain_attr,
         Derivatives<U> {});

      MultiVector MX{ X, setup.N };
      auto dRdU = dop.GetDerivative(U, MX);

      Vector dfem_D(tvsize), mfem_D(tvsize);
      dRdU->AssembleDiagonal(dfem_D);
      blf_pa.AssembleDiagonal(mfem_D);
      REQUIRE(mfem_D.Normlinf() > 1e-8); // guard against comparing zeros
      REQUIRE(HdivMaxError(comm, mfem_D, dfem_D) ==
              MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("Assemble sparse matrix")
   {
      ParBilinearForm blf_fa(&pfes);
      AddHdivIntegrators(blf_fa, form, setup.ir);
      blf_fa.Assemble();
      blf_fa.Finalize();

      DifferentiableOperator dop(in_fds, out_fds, setup.pmesh);
      constexpr auto kernels = DerivativeKernels::AssembleMatrix;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs_t {}, outputs_t {}, *setup.ir, setup.all_domain_attr,
         Derivatives<U> {});

      MultiVector MX{ X, setup.N };
      auto dRdU = dop.GetDerivative(U, MX);

      SparseMatrix *A = nullptr;
      dRdU->Assemble(A);
      REQUIRE(A != nullptr);
      REQUIRE(A->Width() == blf_fa.SpMat().Width());
      TestSameMatrices(*A, blf_fa.SpMat());
      delete A;
   }
}

// ────────────────────────────────────────────────────────────────────────────
template <int DIM>
void hdiv_mass(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   HdivSetup setup(filename, DIM, p);

   static constexpr int U = 0, Coords = 1;
   using IT = Inputs<Value<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Value<U>>;

   CheckHdivOperator<DIM, IT, OT>(setup, hdiv_mass_qf<DIM> {}, HdivForm::Mass);
}

// ────────────────────────────────────────────────────────────────────────────
template <int DIM>
void hdiv_divdiv(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   HdivSetup setup(filename, DIM, p);

   static constexpr int U = 0, Coords = 1;
   using IT = Inputs<Div<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Div<U>>;

   CheckHdivOperator<DIM, IT, OT>(setup, hdiv_divdiv_qf<DIM> {}, HdivForm::DivDiv);
}

// ────────────────────────────────────────────────────────────────────────────
template <int DIM>
void hdiv_mass_divdiv(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   HdivSetup setup(filename, DIM, p);

   static constexpr int U = 0, Coords = 1;
   using IT = Inputs<Value<U>, Div<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Value<U>, Div<U>>;

   CheckHdivOperator<DIM, IT, OT>(setup, hdiv_mass_divdiv_qf<DIM> {},
                                  HdivForm::MassDivDiv);
}

template <int DIM>
void hdiv_mixed_assembly(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   HdivSetup setup(filename, DIM, p);
   H1_FECollection h1_fec(p + 1, DIM);
   ParFiniteElementSpace h1_fes(&setup.pmesh, &h1_fec);

   static constexpr int U = 0, P = 1;
   const auto assemble_and_check = [&](auto qf, auto inputs, auto outputs,
                                       ParFiniteElementSpace &trial_fes,
                                       ParFiniteElementSpace &test_fes,
                                       BilinearFormIntegrator *integrator)
   {
      integrator->SetIntRule(setup.ir);
      ParMixedBilinearForm reference(&trial_fes, &test_fes);
      reference.AddDomainIntegrator(integrator);
      reference.Assemble();
      reference.Finalize();

      DifferentiableOperator dop(
         std::vector{FieldDescriptor{inputs.GetFieldId(), &trial_fes}},
         std::vector{FieldDescriptor{outputs.GetFieldId(), &test_fes}},
         setup.pmesh);
      constexpr auto kernels = DerivativeKernels::AssembleMatrix;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, tuple{inputs, Weight{}}, tuple{outputs}, *setup.ir,
         setup.all_domain_attr, Derivatives<inputs.GetFieldId()> {});

      Vector X(trial_fes.GetTrueVSize());
      X.Randomize(3);
      MultiVector MX{X};
      auto derivative = dop.GetDerivative(inputs.GetFieldId(), MX);

      SparseMatrix *A = nullptr;
      derivative->Assemble(A);
      REQUIRE(A != nullptr);
      REQUIRE(A->Height() == reference.SpMat().Height());
      REQUIRE(A->Width() == reference.SpMat().Width());
      TestSameMatrices(*A, reference.SpMat());
      TestSameMatrices(reference.SpMat(), *A);
      delete A;
   };

   SECTION("RT trial, H1 test")
   {
      assemble_and_check(hdiv_to_h1_qf {}, Div<U> {}, Value<P> {}, setup.pfes,
                         h1_fes, new VectorFEDivergenceIntegrator());
   }

   SECTION("H1 trial, RT test")
   {
      assemble_and_check(h1_to_hdiv_qf {}, Value<P> {}, Div<U> {}, h1_fes,
                         setup.pfes,
                         new TransposeIntegrator(
                            new VectorFEDivergenceIntegrator()));
   }

}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM H(div) 2D", "[Parallel][dFEM][VectorFE]")
{
   const auto p = GenAll({ 0 }, { 1, 2 });
   const auto meshs = { "../../data/inline-quad.mesh" };
   const auto extra = { "../../data/star.mesh", "../../data/rt-2d-q3.mesh" };

   SECTION("Mass") { hdiv_mass<2>(GenAll(meshs, extra), p); }
   SECTION("DivDiv") { hdiv_divdiv<2>(GenAll(meshs, extra), p); }
   SECTION("Mass+DivDiv") { hdiv_mass_divdiv<2>(GenAll(meshs, extra), p); }
   SECTION("Mixed assembly") { hdiv_mixed_assembly<2>(GenAll(meshs, extra), p); }
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM H(div) 3D", "[Parallel][dFEM][VectorFE]")
{
   const auto p = GenAll({ 0 }, { 1 });
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh" };

   SECTION("Mass") { hdiv_mass<3>(GenAll(meshs, extra), p); }
   SECTION("DivDiv") { hdiv_divdiv<3>(GenAll(meshs, extra), p); }
   SECTION("Mass+DivDiv") { hdiv_mass_divdiv<3>(GenAll(meshs, extra), p); }
   SECTION("Mixed assembly") { hdiv_mixed_assembly<3>(GenAll(meshs, extra), p); }
}

#endif // MFEM_USE_MPI

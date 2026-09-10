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
template <int DIM, typename inputs_t, typename outputs_t, typename qf_t,
          typename add_reference_t>
void CheckHdivOperator(HdivSetup &setup, qf_t qf, add_reference_t add_ref)
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
   add_ref(blf_pa);
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

   CheckHdivOperator<DIM, IT, OT>(
      setup, hdiv_mass_qf<DIM> {},
      [](ParBilinearForm &blf)
   { blf.AddDomainIntegrator(new VectorFEMassIntegrator()); });
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

   CheckHdivOperator<DIM, IT, OT>(
      setup, hdiv_divdiv_qf<DIM> {},
      [](ParBilinearForm &blf)
   { blf.AddDomainIntegrator(new DivDivIntegrator()); });
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

   CheckHdivOperator<DIM, IT, OT>(
      setup, hdiv_mass_divdiv_qf<DIM> {},
      [](ParBilinearForm &blf)
   {
      blf.AddDomainIntegrator(new VectorFEMassIntegrator());
      blf.AddDomainIntegrator(new DivDivIntegrator());
   });
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
}

#endif // MFEM_USE_MPI

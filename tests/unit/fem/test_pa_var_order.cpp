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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cmath>
#include <memory>

namespace mfem
{

namespace
{

Mesh MakeVarOrderMesh(int dim, bool nonconforming)
{
   Mesh mesh = (dim == 1) ? Mesh::MakeCartesian1D(6, 1.0)
               : ((dim == 2) ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
                  : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON));
   mesh.EnsureNCMesh(true);
   if (nonconforming && dim > 1)
   {
      Array<int> ref;
      ref.Append(0);
      mesh.GeneralRefinement(ref);
   }
   Vector c(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      mesh.SetAttribute(e, (c(0) < 0.5) ? 1 : 2);
   }
   mesh.SetAttributes();
   return mesh;
}

void AssignMixedOrders(FiniteElementSpace &fes, int min_order, int max_order)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   Vector c(dim);
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      mesh->GetElementCenter(e, c);
      int ord = min_order;
      if (c(0) > 0.6) { ord = max_order; }
      else if (dim > 1 && c(1) > 0.5) { ord = std::min(min_order + 1, max_order); }
      fes.SetElementOrder(e, ord);
   }
   fes.Update(false);
}

void CheckPAvsFA(BilinearForm &a_fa, BilinearForm &a_pa,
                 const FiniteElementSpace &fes,
                 bool check_transpose, bool check_diag)
{
   if (check_diag)
   {
      Vector d_fa_L(fes.GetVSize()), d_ref(fes.GetTrueVSize()),
             d_pa(fes.GetTrueVSize());
      a_fa.SpMat().GetDiag(d_fa_L);
      if (const SparseMatrix *cP = fes.GetConformingProlongation())
      {
         cP->AbsMultTranspose(d_fa_L, d_ref);
      }
      else
      {
         d_ref = d_fa_L;
      }
      a_pa.AssembleDiagonal(d_pa);
      d_pa -= d_ref;
      REQUIRE(d_pa.Norml2() <= 1e-11 * std::max(real_t(1.0), d_ref.Norml2()));
   }

   Array<int> ess_tdof_list;
   if (check_transpose)
   {
      Array<int> ess_bdr(fes.GetMesh()->bdr_attributes.Max());
      ess_bdr = 0;
      if (ess_bdr.Size() > 0) { ess_bdr[0] = 1; }
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   a_fa.SetDiagonalPolicy(Operator::DIAG_ONE);
   OperatorPtr A_fa, A_pa;
   a_fa.FormSystemMatrix(ess_tdof_list, A_fa);
   a_pa.FormSystemMatrix(ess_tdof_list, A_pa);

   const int n = fes.GetTrueVSize();
   Vector x(n), y_fa(n), y_pa(n);
   x.Randomize(17);

   A_fa->Mult(x, y_fa);
   A_pa->Mult(x, y_pa);
   y_pa -= y_fa;
   REQUIRE(y_pa.Norml2() <= 1e-11 * std::max(real_t(1.0), y_fa.Norml2()));

   if (check_transpose)
   {
      A_fa->MultTranspose(x, y_fa);
      A_pa->MultTranspose(x, y_pa);
      y_pa -= y_fa;
      REQUIRE(y_pa.Norml2() <= 1e-11 * std::max(real_t(1.0), y_fa.Norml2()));
   }

   Vector x_pos(n), y_abs(n);
   x_pos = 1.0;
   A_pa->AbsMult(x_pos, y_abs);
   REQUIRE(y_abs.CheckFinite() == 0);
   REQUIRE(y_abs.Norml2() > 0.0);
}

} // namespace

TEST_CASE("VarOrderElementRestriction operators and group ordering",
          "[PartialAssembly][FiniteElementSpace]")
{
   Mesh mesh = MakeVarOrderMesh(3, true);
   const int vdim = GENERATE(1, 2);
   const Ordering::Type ord_type = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   const ElementDofOrdering e_ord =
      GENERATE(ElementDofOrdering::LEXICOGRAPHIC, ElementDofOrdering::NATIVE);
   CAPTURE(vdim, ord_type, e_ord);

   H1_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec, vdim, ord_type);
   AssignMixedOrders(fes, 1, 3);

   const auto *vr = dynamic_cast<const VarOrderElementRestriction*>(
                       fes.GetElementRestriction(e_ord));
   REQUIRE(vr != nullptr);
   REQUIRE(vr->GetNumGroups() >= 2);
   for (int g = 1; g < vr->GetNumGroups(); ++g)
   {
      const bool asc =
         (vr->GetGroupGeometry(g - 1) < vr->GetGroupGeometry(g)) ||
         (vr->GetGroupGeometry(g - 1) == vr->GetGroupGeometry(g) &&
          vr->GetGroupOrder(g - 1) < vr->GetGroupOrder(g));
      REQUIRE(asc);
   }

   Vector x(vr->Width()), y(vr->Height()), z1(vr->Width()), z2(vr->Width());
   x.Randomize(3);
   y.Randomize(5);
   vr->Mult(x, y);
   Vector r(vr->Height());
   r.Randomize(7);
   vr->MultTranspose(r, z1);
   REQUIRE(InnerProduct(y, r) == MFEM_Approx(InnerProduct(x, z1), 1e-11));

   z2 = x;
   vr->AddMultTranspose(r, z2, 2.5);
   z2 -= x;
   z2.Add(-2.5, z1);
   REQUIRE(z2.Norml2() < 1e-12);

   Vector x_abs(x), y_abs(vr->Height()), r_abs(r), z_abs(vr->Width());
   x_abs.Abs();
   r_abs.Abs();
   vr->AbsMult(x_abs, y_abs);
   vr->AbsMultTranspose(r_abs, z_abs);
   REQUIRE(InnerProduct(y_abs, r_abs) ==
           MFEM_Approx(InnerProduct(x_abs, z_abs), 1e-11));
}

TEST_CASE("Variable-order PA for MassIntegrator and DiffusionIntegrator",
          "[PartialAssembly][FiniteElementSpace]")
{
   SECTION("2D and 3D variable-order H1 Diffusion + Mass with coefficients and markers")
   {
      const int dim = GENERATE(2, 3);
      const bool nc = GENERATE(false, true);
      const int coeff_case = GENERATE(0, 1, 2, 3); // scalar, vector, sym mat, nonsym mat
      CAPTURE(dim, nc, coeff_case);

      Mesh mesh = MakeVarOrderMesh(dim, nc);
      H1_FECollection fec(1, dim);
      FiniteElementSpace fes(&mesh, &fec);
      AssignMixedOrders(fes, 1, 3);

      FunctionCoefficient q_mass([](const Vector &p)
      {
         return 1.2 + 0.3 * p(0) + 0.2 * p(1);
      });
      FunctionCoefficient q_diff([](const Vector &p)
      {
         return 2.0 + 0.5 * p(0) * p(0);
      });
      VectorFunctionCoefficient vq(dim, [dim](const Vector &p, Vector &v)
      {
         v.SetSize(dim);
         for (int d = 0; d < dim; ++d) { v(d) = 1.0 + 0.3 * (d + 1) + 0.2 * p(d); }
      });
      MatrixFunctionCoefficient mq_sym(dim, [dim](const Vector &p, DenseMatrix &M)
      {
         M.SetSize(dim, dim);
         M = 0.0;
         for (int i = 0; i < dim; ++i) { M(i, i) = 2.0 + 0.2 * i + 0.1 * p(i); }
         M(0, 1) = M(1, 0) = 0.25 + 0.05 * p(0);
      });
      MatrixFunctionCoefficient mq_asym(dim, [dim](const Vector &p, DenseMatrix &M)
      {
         M.SetSize(dim, dim);
         M = 0.0;
         for (int i = 0; i < dim; ++i) { M(i, i) = 2.0 + 0.2 * i; }
         M(0, 1) = 0.3 + 0.05 * p(0);
         M(1, 0) = -0.15 + 0.04 * p(1);
      });
      DenseSymmetricMatrix smat(dim);
      smat = 0.0;
      for (int i = 0; i < dim; ++i) { smat(i, i) = 2.5 + 0.3 * i; }
      smat(0, 1) = 0.35;
      SymmetricMatrixConstantCoefficient smq(smat);

      Array<int> marker(2);
      marker[0] = 1; marker[1] = 0;

      BilinearForm a_fa(&fes), a_pa(&fes);
      a_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      auto add_diff = [&](BilinearForm &a)
      {
         if (coeff_case == 0) { a.AddDomainIntegrator(new DiffusionIntegrator(q_diff)); }
         else if (coeff_case == 1) { a.AddDomainIntegrator(new DiffusionIntegrator(vq)); }
         else if (coeff_case == 2) { a.AddDomainIntegrator(new DiffusionIntegrator(smq)); }
         else { a.AddDomainIntegrator(new DiffusionIntegrator(mq_asym)); }
         a.AddDomainIntegrator(new MassIntegrator(q_mass), marker);
      };
      add_diff(a_fa);
      add_diff(a_pa);
      a_fa.Assemble(); a_fa.Finalize();
      a_pa.Assemble();

      const bool is_sym = (coeff_case != 3);
      CheckPAvsFA(a_fa, a_pa, fes, is_sym, true);
   }
}

TEST_CASE("Variable-order PA for CurlCurlIntegrator and VectorFEMassIntegrator",
          "[PartialAssembly][FiniteElementSpace]")
{
   const int dim = GENERATE(2, 3);
   const bool nc = GENERATE(false, true);
   const int space_kind = GENERATE(0, 1, 2, 3); // ND nodal, ND IntegratedGLL, RT nodal, RT IntegratedGLL
   CAPTURE(dim, nc, space_kind);

   Mesh mesh = MakeVarOrderMesh(dim, nc);
   std::unique_ptr<FiniteElementCollection> fec;
   if (space_kind == 0)
   {
      fec.reset(new ND_FECollection(1, dim));
   }
   else if (space_kind == 1)
   {
      fec.reset(new ND_FECollection(1, dim, BasisType::GaussLobatto,
                                    BasisType::IntegratedGLL));
   }
   else if (space_kind == 2)
   {
      fec.reset(new RT_FECollection(0, dim));
   }
   else
   {
      fec.reset(new RT_FECollection(0, dim, BasisType::GaussLobatto,
                                    BasisType::IntegratedGLL));
   }
   FiniteElementSpace fes(&mesh, fec.get());
   AssignMixedOrders(fes, 1, 3);

   FunctionCoefficient q_curl([](const Vector &p)
   {
      return 1.0 + 0.25 * p(0) + 0.15 * p(1);
   });
   VectorFunctionCoefficient diag_coeff(dim, [dim](const Vector &p, Vector &v)
   {
      v.SetSize(dim);
      for (int d = 0; d < dim; ++d) { v(d) = 1.5 + 0.3 * d + 0.1 * p(d); }
   });

   MatrixFunctionCoefficient mq_asym(dim, [dim](const Vector &p, DenseMatrix &M)
   {
      M.SetSize(dim, dim);
      M = 0.0;
      for (int i = 0; i < dim; ++i) { M(i, i) = 2.0 + 0.2 * i; }
      M(0, 1) = 0.25 + 0.03 * p(0);
      M(1, 0) = -0.1 + 0.02 * p(1);
   });

   Array<int> marker(2);
   marker[0] = 1; marker[1] = 0;

   BilinearForm a_fa(&fes), a_pa(&fes);
   a_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto add_integs = [&](BilinearForm &a)
   {
      if (space_kind < 2)
      {
         a.AddDomainIntegrator(new CurlCurlIntegrator(q_curl));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(diag_coeff), marker);
      }
      else
      {
         a.AddDomainIntegrator(new DivDivIntegrator(q_curl));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(diag_coeff), marker);
      }
   };
   add_integs(a_fa);
   add_integs(a_pa);
   a_fa.Assemble(); a_fa.Finalize();
   a_pa.Assemble();
   CheckPAvsFA(a_fa, a_pa, fes, true, true);

   // Asymmetric matrix coefficient check for Mult & MultTranspose on CurlCurl/VectorFEMass.
   if (dim == 3 && space_kind == 0)
   {
      BilinearForm b_fa(&fes), b_pa(&fes);
      b_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      b_fa.AddDomainIntegrator(new CurlCurlIntegrator(mq_asym));
      b_pa.AddDomainIntegrator(new CurlCurlIntegrator(mq_asym));
      b_fa.Assemble(); b_fa.Finalize();
      b_pa.Assemble();
      CheckPAvsFA(b_fa, b_pa, fes, false, true);
   }
}

#ifdef MFEM_USE_MPI
TEST_CASE("Parallel variable-order PA matches FA for H1, ND, and RT",
          "[PartialAssembly][FiniteElementSpace][Parallel]")
{
   int nranks = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);

   const int kind = GENERATE(0, 1, 2, 3); // H1, ND IntegratedGLL, RT nodal, RT IntegratedGLL
   CAPTURE(kind);

   Mesh serial = MakeVarOrderMesh(3, true);
   Array<int> part(serial.GetNE());
   for (int e = 0; e < serial.GetNE(); ++e) { part[e] = e % nranks; }

   ParMesh pmesh(MPI_COMM_WORLD, serial, part.GetData());
   std::unique_ptr<FiniteElementCollection> fec;
   if (kind == 0) { fec.reset(new H1_FECollection(1, 3)); }
   else if (kind == 1)
   {
      fec.reset(new ND_FECollection(1, 3, BasisType::GaussLobatto,
                                    BasisType::IntegratedGLL));
   }
   else if (kind == 2) { fec.reset(new RT_FECollection(0, 3)); }
   else
   {
      fec.reset(new RT_FECollection(0, 3, BasisType::GaussLobatto,
                                    BasisType::IntegratedGLL));
   }

   ParFiniteElementSpace pfes(&pmesh, fec.get());
   AssignMixedOrders(pfes, 1, 3);

   FunctionCoefficient q([](const Vector &p)
   {
      return 1.0 + 0.3 * p(0) + 0.2 * p(2);
   });
   Array<int> marker(2);
   marker[0] = 1; marker[1] = 0;

   ParBilinearForm a_fa(&pfes), a_pa(&pfes);
   a_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto add_ops = [&](ParBilinearForm &a)
   {
      if (kind == 0)
      {
         a.AddDomainIntegrator(new DiffusionIntegrator(q));
         a.AddDomainIntegrator(new MassIntegrator(q), marker);
      }
      else if (kind == 1)
      {
         a.AddDomainIntegrator(new CurlCurlIntegrator(q));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(q), marker);
      }
      else
      {
         a.AddDomainIntegrator(new DivDivIntegrator(q));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(q), marker);
      }
   };
   add_ops(a_fa);
   add_ops(a_pa);
   a_fa.Assemble(); a_fa.Finalize();
   a_pa.Assemble();

   const int n = pfes.GetTrueVSize();
   Vector d_fa_L(pfes.GetVSize()), d_ref(n), d_pa(n);
   a_fa.SpMat().GetDiag(d_fa_L);
   pfes.Dof_TrueDof_Matrix()->AbsMultTranspose(d_fa_L, d_ref);
   a_pa.AssembleDiagonal(d_pa);
   d_pa -= d_ref;
   real_t dloc[2] = { d_pa * d_pa, d_ref * d_ref };
   real_t dglob[2] = { 0.0, 0.0 };
   MPI_Allreduce(dloc, dglob, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(std::sqrt(dglob[0]) <= 1e-11 * std::max(real_t(1.0), std::sqrt(dglob[1])));

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   if (ess_bdr.Size() > 0) { ess_bdr[0] = 1; }
   pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   OperatorPtr A_fa, A_pa;
   a_fa.FormSystemMatrix(ess_tdof_list, A_fa);
   a_pa.FormSystemMatrix(ess_tdof_list, A_pa);

   Vector x(n), y_fa(n), y_pa(n);
   x.Randomize(31);
   A_fa->Mult(x, y_fa);
   A_pa->Mult(x, y_pa);
   y_pa -= y_fa;
   real_t loc[2] = { y_pa * y_pa, y_fa * y_fa };
   real_t glob[2] = { 0.0, 0.0 };
   MPI_Allreduce(loc, glob, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   REQUIRE(std::sqrt(glob[0]) <= 1e-11 * std::max(real_t(1.0), std::sqrt(glob[1])));
}
#endif

} // namespace mfem

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
constexpr real_t a_coef = 1.0;
constexpr real_t b_coef = 2.0;
constexpr real_t c_coef = 3.0;
constexpr real_t omega_val = 10.0;

real_t V_exact_fn(const Vector &x)
{
   return a_coef*x[0] + b_coef*x[1] + c_coef*x[2];
}
} // namespace

TEST_CASE("Mixed Sesquilinear Form", "[MixedSesquilinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian3D(10, 10, 1, Element::HEXAHEDRON);
   H1_FECollection fec_h1(1, mesh.Dimension());
   FiniteElementSpace fespace_h1(&mesh, &fec_h1);
   ND_FECollection fec_nd(1, mesh.Dimension());
   FiniteElementSpace fespace_nd(&mesh, &fec_nd);

   Array<int> dbc_bdr(mesh.bdr_attributes.Max());
   dbc_bdr = 1;
   Array<int> ess_tdof_list_h1;
   Array<int> ess_tdof_list_nd;
   fespace_h1.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list_h1);
   fespace_nd.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list_nd);

   ConstantCoefficient omega(omega_val);
   ConstantCoefficient neg_omega(-omega_val);
   FunctionCoefficient V_exact_real(V_exact_fn);
   Vector A_imag_vec({a_coef/omega_val, b_coef/omega_val, c_coef/omega_val});
   VectorConstantCoefficient A_exact_imag(A_imag_vec);

   ComplexGridFunction V(&fespace_h1);
   ComplexGridFunction A(&fespace_nd);

   V = 0.0;
   A = 0.0;
   V.real().ProjectBdrCoefficient(V_exact_real, dbc_bdr);
   A.imag().ProjectBdrCoefficientTangent(A_exact_imag, dbc_bdr);

   ComplexLinearForm b_h1(&fespace_h1);
   b_h1 = 0.0;
   b_h1.Assemble();
   ComplexLinearForm b_nd(&fespace_nd);
   b_nd = 0.0;
   b_nd.Assemble();

   // Add integrators to the blocks
   MixedSesquilinearForm a_h1_nd(&fespace_h1, &fespace_nd);
   a_h1_nd.AddDomainIntegrator(new MixedVectorGradientIntegrator, nullptr);
   a_h1_nd.Assemble();

   MixedSesquilinearForm a_nd_h1(&fespace_nd, &fespace_h1);
   a_nd_h1.AddDomainIntegrator(nullptr,
                               new MixedVectorWeakDivergenceIntegrator(neg_omega));
   a_nd_h1.Assemble();

   SesquilinearForm a_h1(&fespace_h1);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator, nullptr);
   a_h1.Assemble();

   SesquilinearForm a_nd(&fespace_nd);
   a_nd.AddDomainIntegrator(new CurlCurlIntegrator, nullptr);
   a_nd.AddDomainIntegrator(nullptr, new VectorFEMassIntegrator(omega));
   a_nd.Assemble();

   // Set block offsets (doubled for real+imag)
   mfem::Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = 2 * fespace_h1.GetTrueVSize();
   bOffsets[2] = 2 * fespace_nd.GetTrueVSize();
   bOffsets.PartialSum();

   OperatorPtr A_h1, A_nd, A_h1_nd, A_nd_h1;
   BlockVector trueX(bOffsets), trueRHS(bOffsets);
   Vector B_h1, X_h1, B_nd, X_nd;

   trueX = 0.0;
   trueRHS = 0.0;

   // Form the diagonal entries
   a_h1.FormLinearSystem(ess_tdof_list_h1, V, b_h1, A_h1, X_h1, B_h1);
   a_nd.FormLinearSystem(ess_tdof_list_nd, A, b_nd, A_nd, X_nd, B_nd);

   trueX.GetBlock(0) = X_h1;
   trueX.GetBlock(1) = X_nd;
   trueRHS.GetBlock(0) += B_h1;
   trueRHS.GetBlock(1) += B_nd;

   // Form the off-diagonal entries
   a_h1_nd.FormRectangularLinearSystem(ess_tdof_list_h1, ess_tdof_list_nd, V, b_nd,
                                       A_h1_nd, X_h1, B_nd);
   a_nd_h1.FormRectangularLinearSystem(ess_tdof_list_nd, ess_tdof_list_h1, A, b_h1,
                                       A_nd_h1, X_nd, B_h1);

   trueRHS.GetBlock(0) += B_h1;
   trueRHS.GetBlock(1) += B_nd;

   auto *Ah1 = A_h1.As<ComplexSparseMatrix>();
   auto *And = A_nd.As<ComplexSparseMatrix>();
   auto *Ah1nd = A_h1_nd.As<ComplexSparseMatrix>();
   auto *Andh1 = A_nd_h1.As<ComplexSparseMatrix>();

   BlockOperator blockOp(bOffsets);
   blockOp.SetBlock(0, 0, Ah1);
   blockOp.SetBlock(1, 1, And);
   blockOp.SetBlock(0, 1, Andh1);
   blockOp.SetBlock(1, 0, Ah1nd);

   SparseMatrix *Sh1 = Ah1->GetSystemMatrix();
   SparseMatrix *Snd = And->GetSystemMatrix();
   GSSmoother smoothSh1(*Sh1), smoothSnd(*Snd);

   BlockDiagonalPreconditioner P(bOffsets);
   P.SetDiagonalBlock(0, &smoothSh1);
   P.SetDiagonalBlock(1, &smoothSnd);

   GMRESSolver gmres;
   gmres.SetOperator(blockOp);
   gmres.SetPreconditioner(P);
   gmres.SetAbsTol(1e-10);
   gmres.SetMaxIter(2000);
   gmres.SetKDim(200);
   gmres.SetPrintLevel(2);
   gmres.Mult(trueRHS, trueX);

   std::fprintf(stderr,
                "[GMRES] converged=%d iters=%d final_norm=%.3e\n",
                gmres.GetConverged(), gmres.GetNumIterations(),
                gmres.GetFinalNorm());
   std::fprintf(stderr, "trueX block sizes: %d, %d  (matrix N=%d)\n",
                trueX.GetBlock(0).Size(), trueX.GetBlock(1).Size(),
                blockOp.Height());

   delete Sh1;
   delete Snd;

   V = trueX.GetBlock(0);
   A = trueX.GetBlock(1);

   // Check solution
   ConstantCoefficient zero(0.0);
   VectorConstantCoefficient zero_vec(Vector({0.0, 0.0, 0.0}));

   real_t err_Vr = V.real().ComputeL2Error(V_exact_real);
   real_t err_Vi = V.imag().ComputeL2Error(zero);
   real_t err_Ar = A.real().ComputeL2Error(zero_vec);
   real_t err_Ai = A.imag().ComputeL2Error(A_exact_imag);
   
   REQUIRE(err_Vr == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Vi == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Ar == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Ai == MFEM_Approx(0.0, 1e-5));
}

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_SUPERLU

TEST_CASE("Parallel Mixed Sesquilinear Form",
          "[MixedSesquilinearForm][Parallel]")
{
   Mesh mesh = Mesh::MakeCartesian3D(10, 10, 1, Element::HEXAHEDRON);
   ParMesh par_mesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec_h1(1, mesh.Dimension());
   ParFiniteElementSpace fespace_h1(&par_mesh, &fec_h1);
   ND_FECollection fec_nd(1, mesh.Dimension());
   ParFiniteElementSpace fespace_nd(&par_mesh, &fec_nd);

   Array<int> dbc_bdr(par_mesh.bdr_attributes.Max());
   dbc_bdr = 1;
   Array<int> ess_tdof_list_h1;
   Array<int> ess_tdof_list_nd;
   fespace_h1.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list_h1);
   fespace_nd.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list_nd);

   ConstantCoefficient omega(omega_val);
   ConstantCoefficient neg_omega(-omega_val);
   FunctionCoefficient V_exact_real(V_exact_fn);
   Vector A_imag_vec({a_coef/omega_val, b_coef/omega_val, c_coef/omega_val});
   VectorConstantCoefficient A_exact_imag(A_imag_vec);

   ParComplexGridFunction V(&fespace_h1);
   ParComplexGridFunction A(&fespace_nd);

   V = 0.0;
   A = 0.0;
   V.real().ProjectBdrCoefficient(V_exact_real, dbc_bdr);
   A.imag().ProjectBdrCoefficientTangent(A_exact_imag, dbc_bdr);

   ParComplexLinearForm b_h1(&fespace_h1);
   b_h1 = 0.0;
   b_h1.Assemble();
   ParComplexLinearForm b_nd(&fespace_nd);
   b_nd = 0.0;
   b_nd.Assemble();

   // Add integrators to the blocks
   ParMixedSesquilinearForm a_h1_nd(&fespace_h1, &fespace_nd);
   a_h1_nd.AddDomainIntegrator(new MixedVectorGradientIntegrator, nullptr);
   a_h1_nd.Assemble();

   ParMixedSesquilinearForm a_nd_h1(&fespace_nd, &fespace_h1);
   a_nd_h1.AddDomainIntegrator(nullptr,
                               new MixedVectorWeakDivergenceIntegrator(neg_omega));
   a_nd_h1.Assemble();

   ParSesquilinearForm a_h1(&fespace_h1);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator, nullptr);
   a_h1.Assemble();

   ParSesquilinearForm a_nd(&fespace_nd);
   a_nd.AddDomainIntegrator(new CurlCurlIntegrator, nullptr);
   a_nd.AddDomainIntegrator(nullptr, new VectorFEMassIntegrator(omega));
   a_nd.Assemble();

   mfem::Array2D<const mfem::HypreParMatrix *> h_blocks;
   h_blocks.SetSize(2, 2);
   h_blocks = nullptr;

   // Set block offsets
   mfem::Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = 2 * fespace_h1.TrueVSize();
   bOffsets[2] = 2 * fespace_nd.TrueVSize();
   bOffsets.PartialSum();

   OperatorPtr A_h1, A_nd, A_h1_nd, A_nd_h1;
   BlockVector trueX(bOffsets), trueRHS(bOffsets);
   Vector B_h1, X_h1, B_nd, X_nd;

   trueX = 0.0;
   trueRHS = 0.0;

   // Form the diagonal entries
   a_h1.FormLinearSystem(ess_tdof_list_h1, V, b_h1, A_h1, X_h1, B_h1);
   a_nd.FormLinearSystem(ess_tdof_list_nd, A, b_nd, A_nd, X_nd, B_nd);

   trueX.GetBlock(0) = X_h1;
   trueX.GetBlock(1) = X_nd;
   trueRHS.GetBlock(0) += B_h1;
   trueRHS.GetBlock(1) += B_nd;

   // Form the off-diagonal entries
   a_h1_nd.FormRectangularLinearSystem(ess_tdof_list_h1, ess_tdof_list_nd, V, b_nd,
                                       A_h1_nd, X_h1, B_nd);
   a_nd_h1.FormRectangularLinearSystem(ess_tdof_list_nd, ess_tdof_list_h1, A, b_h1,
                                       A_nd_h1, X_nd, B_h1);

   trueRHS.GetBlock(0) += B_h1;
   trueRHS.GetBlock(1) += B_nd;

   h_blocks(0,0) = A_h1.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   h_blocks(1,1) = A_nd.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   h_blocks(0,1) = A_nd_h1.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   h_blocks(1,0) = A_h1_nd.As<ComplexHypreParMatrix>()->GetSystemMatrix();

   OperatorHandle op(HypreParMatrixFromBlocks(h_blocks));

   SuperLURowLocMatrix S_op(*op);
   SuperLUSolver superlu(MPI_COMM_WORLD);
   superlu.SetPrintStatistics(false);
   superlu.SetSymmetricPattern(false);
   superlu.SetOperator(S_op);
   superlu.Mult(trueRHS, trueX);

   trueX.GetBlock(0).SyncAliasMemory(trueX);
   trueX.GetBlock(1).SyncAliasMemory(trueX);

   V.Distribute(trueX.GetBlock(0));
   A.Distribute(trueX.GetBlock(1));

   // Check solution
   ConstantCoefficient zero(0.0);
   VectorConstantCoefficient zero_vec(Vector({0.0, 0.0, 0.0}));

   real_t err_Vr = V.real().ComputeL2Error(V_exact_real);
   real_t err_Vi = V.imag().ComputeL2Error(zero);
   real_t err_Ar = A.real().ComputeL2Error(zero_vec);
   real_t err_Ai = A.imag().ComputeL2Error(A_exact_imag);
   
   REQUIRE(err_Vr == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Vi == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Ar == MFEM_Approx(0.0, 1e-5));
   REQUIRE(err_Ai == MFEM_Approx(0.0, 1e-5));
}

#endif
#endif



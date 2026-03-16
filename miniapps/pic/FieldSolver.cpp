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

#include "FieldSolver.hpp"

#include <algorithm>
#include <iostream>
#include <memory>

#include "ParticleMover.hpp"

namespace
{
   constexpr mfem::real_t EPSILON = 1.0;

   mfem::HypreParMatrix* MakeDiagonalMatrix(
      mfem::Vector& diag, const mfem::ParFiniteElementSpace& fes)
   {
      const int n = diag.Size();

      mfem::SparseMatrix diag_spmat;
      diag_spmat.OverrideSize(n, n);
      diag_spmat.GetMemoryI().New(n + 1, mfem::Device::GetDeviceMemoryType());
      diag_spmat.GetMemoryJ().New(n, mfem::Device::GetDeviceMemoryType());
      diag_spmat.GetMemoryData().New(n, mfem::Device::GetDeviceMemoryType());

      {
         int* I = diag_spmat.WriteI();
         int* J = diag_spmat.WriteJ();
         mfem::real_t* A = diag_spmat.WriteData();
         const mfem::real_t* d_diag = diag.Read();
         MFEM_FORALL(i, n + 1, I[i] = i;);
         MFEM_FORALL(i, n,
         {
            J[i] = i;
            A[i] = d_diag[i];
         });
      }

      HYPRE_BigInt global_size = fes.GlobalTrueVSize();
      HYPRE_BigInt* row_starts = fes.GetTrueDofOffsets();
      mfem::HypreParMatrix D(fes.GetComm(), global_size, row_starts, &diag_spmat);
      return new mfem::HypreParMatrix(D);
   }

   mfem::real_t ComputeGlobalSum(mfem::ParLinearForm& lf)
   {
      std::unique_ptr<mfem::HypreParVector> lf_true(lf.ParallelAssemble());
      const mfem::real_t local_sum = lf_true->Sum();
      mfem::real_t global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM,
                    lf.ParFESpace()->GetComm());
      return global_sum;
   }
}

using namespace std;
using namespace mfem;
using namespace mfem::common;

FieldSolver::FieldSolver(ParFiniteElementSpace* phi_fes,
                         ParFiniteElementSpace* E_fes,
                         FindPointsGSLIB& E_finder_, real_t diffusivity,
                         bool precompute_neutralizing_const_)
    : precompute_neutralizing_const(precompute_neutralizing_const_),
      E_finder(E_finder_),
      b(phi_fes)
{
   ParMesh* pmesh = phi_fes->GetParMesh();
   real_t local_domain_volume = 0.0;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      local_domain_volume += pmesh->GetElementVolume(i);
   }
   MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                 phi_fes->GetParMesh()->GetComm());

   {
      ParBilinearForm dm(phi_fes);
      ConstantCoefficient epsilon(EPSILON);
      dm.AddDomainIntegrator(new DiffusionIntegrator(epsilon));
      dm.Assemble();
      dm.Finalize();
      diffusion_matrix = dm.ParallelAssemble();
   }

   {
      ParBilinearForm dm(phi_fes);
      dm.AddDomainIntegrator(new DiffusionIntegrator);
      dm.Assemble();
      dm.Finalize();
      stiffness_matrix = dm.ParallelAssemble();

      // Build mass matrix M
      ParBilinearForm m(phi_fes);
      m.AddDomainIntegrator(new MassIntegrator());
      m.Assemble();
      m.Finalize();
      mass_matrix = m.ParallelAssemble();

      const int ntrue = phi_fes->GetTrueVSize();
      offsets.SetSize(3);
      offsets[0] = 0;
      offsets[1] = ntrue;
      offsets[2] = 2 * ntrue;

      block_matrix = new BlockOperator(offsets);
      block_matrix->SetBlock(0, 0, mass_matrix);
      block_matrix->SetBlock(0, 1, stiffness_matrix,
                                          diffusivity);
      block_matrix->SetBlock(1, 0, stiffness_matrix, -1.0);
      block_matrix->SetBlock(1, 1, mass_matrix);

      Vector inv_mass_diag;
      mass_matrix->GetDiag(inv_mass_diag);
      inv_mass_diag.Reciprocal();
      std::unique_ptr<HypreParMatrix> inv_mass_diag_matrix(
         MakeDiagonalMatrix(inv_mass_diag, *phi_fes));
      std::unique_ptr<HypreParMatrix> K_Minv_K(
         RAP(inv_mass_diag_matrix.get(), stiffness_matrix));
      schur_matrix =
         Add(1.0, *mass_matrix, diffusivity, *K_Minv_K);

      block_preconditioner =
         new BlockLowerTriangularPreconditioner(offsets);
      block_preconditioner->SetDiagonalBlock(
         0, new HypreDiagScale(*mass_matrix));
      block_preconditioner->SetBlock(
         1, 0, new ScaledOperator(stiffness_matrix, -1.0));
      block_preconditioner->SetDiagonalBlock(
         1, new HypreBoomerAMG(*schur_matrix));
      static_cast<HypreBoomerAMG&>(
         block_preconditioner->GetBlock(1, 1)).SetPrintLevel(0);
      block_preconditioner->owns_blocks = 1;
   }

   {
      grad_interpolator = new ParDiscreteLinearOperator(phi_fes, E_fes);
      grad_interpolator->AddDomainInterpolator(new GradientInterpolator);
      grad_interpolator->Assemble();
   }
}

FieldSolver::~FieldSolver()
{
   delete diffusion_matrix;
   delete block_preconditioner;
   delete block_matrix;
   delete schur_matrix;
   delete stiffness_matrix;
   delete mass_matrix;
   delete precomputed_neutralizing_lf;
   delete grad_interpolator;
}

const ParLinearForm& FieldSolver::ComputeNeutralizingRHS(
   ParFiniteElementSpace* pfes, const ParticleVector& Q, MPI_Comm comm)
{
   const int npt = Q.Size();
   const Array<unsigned int>& code = E_finder.GetCode();

   if (!precompute_neutralizing_const || precomputed_neutralizing_lf == nullptr)
   {
      real_t local_sum = 0.0;
      for (int p = 0; p < npt; ++p)
      {
         MFEM_ASSERT(code[p] != 2, "Particle " << p << " not found.");
         local_sum += Q(p);
      }

      real_t global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

      neutralizing_const = -global_sum / domain_volume;
      if (Mpi::Root())
      {
         cout << "Total charge: " << global_sum
              << ", Domain volume: " << domain_volume
              << ", Neutralizing constant: " << neutralizing_const << endl;
         if (precompute_neutralizing_const)
         {
            cout << "Further updates will use this precomputed neutralizing "
                    "constant."
                 << endl;
         }
      }

      delete precomputed_neutralizing_lf;
      precomputed_neutralizing_lf = new ParLinearForm(pfes);
      *precomputed_neutralizing_lf = 0.0;
      ConstantCoefficient neutralizing_coeff(neutralizing_const);
      precomputed_neutralizing_lf->AddDomainIntegrator(
         new DomainLFIntegrator(neutralizing_coeff));
      precomputed_neutralizing_lf->Assemble();
   }
   return *precomputed_neutralizing_lf;
}

void FieldSolver::DepositCharge(ParFiniteElementSpace* pfes,
                                const ParticleVector& Q, ParLinearForm& b)
{
   const int npt = Q.Size();
   ParMesh* pmesh = pfes->GetParMesh();
   const int dim = pmesh->SpaceDimension();
   int curr_rank;
   MPI_Comm_rank(pmesh->GetComm(), &curr_rank);

   // 0: inside, 1: boundary, 2: not found.
   const Array<unsigned int>& code = E_finder.GetCode();
   const Array<unsigned int>& proc = E_finder.GetProc();
   const Array<unsigned int>& elem = E_finder.GetElem();
   const Vector& rref = E_finder.GetReferencePosition();

   Array<int> dofs;

   for (int p = 0; p < npt; ++p)
   {
      MFEM_ASSERT(code[p] != 2, "Particle " << p << " not found.");

      MFEM_ASSERT((int)proc[p] == curr_rank,
                  "Particle " << p << " found in element owned by rank "
                              << proc[p] << " but current rank is " << curr_rank
                              << "." << endl
                              << "You must call redistribute everytime before "
                                 "updating the density grid function.");
      const int e = elem[p];

      IntegrationPoint ip;
      ip.Set(rref.GetData() + dim * p, dim);

      const FiniteElement& fe = *pfes->GetFE(e);
      const int ldofs = fe.GetDof();

      Vector shape(ldofs);
      fe.CalcShape(ip, shape);

      pfes->GetElementDofs(e, dofs);

      const real_t q_p = Q(p);
      b.AddElementVector(dofs, q_p, shape);
   }
}

void FieldSolver::UpdatePhiGridFunction(ParticleSet& particles,
                                        ParGridFunction& phi_gf,
                                        ParGridFunction& rho_gf)
{
   ParFiniteElementSpace* pfes = phi_gf.ParFESpace();

   ParticleVector& Q = particles.Field(ParticleMover::CHARGE);

   MPI_Comm comm = pfes->GetComm();
   b = ComputeNeutralizingRHS(pfes, Q, comm);
   if (Mpi::Root())
   {
      cout << "Total charge A: " << ComputeGlobalSum(b) << "\t";
   }
   else
   {
      ComputeGlobalSum(b);
   }

   DepositCharge(pfes, Q, b);
   if (Mpi::Root())
   {
      cout << "Total charge B: " << ComputeGlobalSum(b) << "\t";
   }
   else
   {
      ComputeGlobalSum(b);
   }

   DiffuseRHS(b, rho_gf);
   if (Mpi::Root())
   {
      cout << "Total charge C: " << ComputeGlobalSum(b) << endl;
   }
   else
   {
      ComputeGlobalSum(b);
   }

   HypreParVector B(pfes);
   b.ParallelAssemble(B);

   phi_gf = 0.0;
   HypreParVector Phi_true(pfes);
   Phi_true = 0.0;

   HyprePCG solver(diffusion_matrix->GetComm());
   solver.SetOperator(*diffusion_matrix);
   solver.SetTol(1e-12);
   solver.SetMaxIter(200);
   solver.SetPrintLevel(0);

   HypreBoomerAMG prec(*diffusion_matrix);
   prec.SetPrintLevel(0);
   solver.SetPreconditioner(prec);

   OrthoSolver ortho(comm);
   ortho.SetSolver(solver);
   ortho.Mult(B, Phi_true);

   phi_gf.Distribute(Phi_true);
}

void FieldSolver::UpdateEGridFunction(ParGridFunction& phi_gf,
                                      ParGridFunction& E_gf)
{
   grad_interpolator->Mult(phi_gf, E_gf);
   E_gf.Neg();
}

void FieldSolver::DiffuseRHS(ParLinearForm& b, ParGridFunction& rho_gf)
{
   HypreParVector* B = b.ParallelAssemble();
   ParFiniteElementSpace* pfes = rho_gf.ParFESpace();
   BlockVector X(offsets);
   BlockVector RHS(offsets);
   X = 0.0;
   RHS = 0.0;
   RHS.GetBlock(0) = *B;

   GMRESSolver solver(pfes->GetComm());
   solver.SetOperator(*block_matrix);
   solver.SetRelTol(1e-12);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(1000);
   solver.SetPrintLevel(0);
   solver.SetPreconditioner(*block_preconditioner);
   solver.Mult(RHS, X);

   rho_gf = 0.0;
   delete B;

   rho_gf.SetFromTrueDofs(X.GetBlock(0));

   b = 0.0;

   b.GetDLFI()->DeleteAll();

   GridFunctionCoefficient rho_coeff(&rho_gf);
   b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
   b.Assemble();
}

real_t FieldSolver::ComputeFieldEnergy(const ParGridFunction& E_gf) const
{
   const ParFiniteElementSpace* fes = E_gf.ParFESpace();
   const ParMesh* pmesh = fes->GetParMesh();

   const int order = fes->GetMaxElementOrder();
   const int qorder = std::max(2, 2 * order + 1);

   const IntegrationRule* irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      irs[g] = &IntRules.Get(g, qorder);
   }

   Vector zero(pmesh->Dimension());
   zero = 0.0;
   VectorConstantCoefficient zero_vec(zero);

   const real_t E_l2 = E_gf.ComputeL2Error(zero_vec, irs);
   return 0.5 * EPSILON * E_l2 * E_l2;
}

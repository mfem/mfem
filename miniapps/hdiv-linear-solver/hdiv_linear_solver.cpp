// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "hdiv_linear_solver.hpp"
#include "discrete_divergence.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

/// Replace x[i] with 1.0/x[i] for all i.
void Reciprocal(Vector &x)
{
   const int n = x.Size();
   real_t *d_x = x.ReadWrite();
   MFEM_FORALL(i, n, d_x[i] = 1.0/d_x[i]; );
}

/// Return a new HypreParMatrix with given diagonal entries
HypreParMatrix *MakeDiagonalMatrix(Vector &diag,
                                   const ParFiniteElementSpace &fes)
{
   const int n = diag.Size();

   SparseMatrix diag_spmat;
   diag_spmat.OverrideSize(n, n);
   diag_spmat.GetMemoryI().New(n+1, Device::GetDeviceMemoryType());
   diag_spmat.GetMemoryJ().New(n, Device::GetDeviceMemoryType());
   diag_spmat.GetMemoryData().New(n, Device::GetDeviceMemoryType());

   {
      int *I = diag_spmat.WriteI();
      int *J = diag_spmat.WriteJ();
      real_t *A = diag_spmat.WriteData();
      const real_t *d_diag = diag.Read();
      MFEM_FORALL(i, n+1, I[i] = i;);
      MFEM_FORALL(i, n,
      {
         J[i] = i;
         A[i] = d_diag[i];
      });
   }

   HYPRE_BigInt global_size = fes.GlobalTrueVSize();
   HYPRE_BigInt *row_starts = fes.GetTrueDofOffsets();
   HypreParMatrix D(fes.GetComm(), global_size, row_starts, &diag_spmat);
   return new HypreParMatrix(D); // make a deep copy
}

const IntegrationRule &GetMassIntRule(FiniteElementSpace &fes_l2)
{
   Mesh *mesh = fes_l2.GetMesh();
   const FiniteElement *fe = fes_l2.GetTypicalFE();
   return MassIntegrator::GetRule(
             *fe, *fe, *mesh->GetTypicalElementTransformation());
}

HdivSaddlePointSolver::HdivSaddlePointSolver(
   ParMesh &mesh, ParFiniteElementSpace &fes_rt_, ParFiniteElementSpace &fes_l2_,
   Coefficient &L_coeff_, Coefficient &R_coeff_, const Array<int> &ess_rt_dofs_,
   Mode mode_)
   : minres(mesh.GetComm()),
     order(fes_rt_.GetMaxElementOrder()),
     fec_l2(order - 1, mesh.Dimension(), b2, mt),
     fes_l2(&mesh, &fec_l2),
     fec_rt(order - 1, mesh.Dimension(), b1, b2),
     fes_rt(&mesh, &fec_rt),
     ess_rt_dofs(ess_rt_dofs_),
     basis_l2(fes_l2_),
     basis_rt(fes_rt_),
     convert_map_type(fes_l2_.GetTypicalFE()->GetMapType() == FiniteElement::VALUE),
     mass_l2(&fes_l2),
     mass_rt(&fes_rt),
     L_coeff(L_coeff_),
     R_coeff(R_coeff_),
     mode(mode_),
     qs(mesh, GetMassIntRule(fes_l2)),
     W_coeff_qf(qs),
     W_mix_coeff_qf(qs),
     W_coeff(W_coeff_qf),
     W_mix_coeff(W_mix_coeff_qf)
{
   // If the user gives zero L coefficient, switch mode to DARCY_ZERO
   auto *L_const_coeff = dynamic_cast<ConstantCoefficient*>(&L_coeff);
   zero_l2_block = (L_const_coeff && L_const_coeff->constant == 0.0);

   if (mode == Mode::GRAD_DIV)
   {
      MFEM_VERIFY(!zero_l2_block,
                  "Mode::GRAD_DIV incompatible with zero coefficient.");
   }

   mass_l2.AddDomainIntegrator(new MassIntegrator(W_coeff));
   mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator(&R_coeff));
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   D.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_rt_dofs));
   Dt.reset(D->Transpose());

   // Versions without BCs needed for elimination
   D_e.reset(FormDiscreteDivergenceMatrix(fes_rt, fes_l2, empty));
   mass_rt.FormSystemMatrix(empty, R_e);

   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_l2.GetTrueVSize();
   offsets[2] = offsets[1] + fes_rt.GetTrueVSize();

   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().None());
   minres.iterative_mode = false;

   R_diag.SetSize(fes_rt.GetTrueVSize());
   L_diag.SetSize(fes_l2.GetTrueVSize());

   S_inv.SetPrintLevel(0);

   if (mode == Mode::DARCY && !zero_l2_block)
   {
      ParBilinearForm mass_l2_unweighted(&fes_l2);
      QuadratureFunction det_J_qf(qs);
      QuadratureFunctionCoefficient det_J_coeff(det_J_qf);
      if (convert_map_type)
      {
         const auto flags = GeometricFactors::DETERMINANTS;
         auto *geom = fes_l2.GetMesh()->GetGeometricFactors(qs.GetIntRule(0), flags);
         det_J_qf = geom->detJ;
         mass_l2_unweighted.AddDomainIntegrator(new MassIntegrator(det_J_coeff));
      }
      else
      {
         mass_l2_unweighted.AddDomainIntegrator(new MassIntegrator);
      }
      mass_l2_unweighted.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mass_l2_unweighted.Assemble();
      const int n_l2 = fes_l2.GetTrueVSize();
      L_diag_unweighted.SetSize(n_l2);
      mass_l2_unweighted.AssembleDiagonal(L_diag_unweighted);
   }

   Setup();
}

HdivSaddlePointSolver::HdivSaddlePointSolver(
   ParMesh &mesh, ParFiniteElementSpace &fes_rt_, ParFiniteElementSpace &fes_l2_,
   Coefficient &R_coeff_, const Array<int> &ess_rt_dofs_)
   : HdivSaddlePointSolver(mesh, fes_rt_, fes_l2_, zero, R_coeff_,
                           ess_rt_dofs_, Mode::DARCY)
{ }

void HdivSaddlePointSolver::Setup()
{
   const auto flags = GeometricFactors::DETERMINANTS;
   auto *geom = fes_l2.GetMesh()->GetGeometricFactors(qs.GetIntRule(0), flags);

   if (!zero_l2_block) { L_coeff.Project(W_coeff_qf); }
   // In "grad-div mode", the transformation matrix is scaled by the coefficient
   // of the mass and divergence matrices.
   // In "Darcy mode", the transformation matrix is unweighted.
   if (mode == Mode::GRAD_DIV) { W_mix_coeff_qf = W_coeff_qf; }
   else { W_mix_coeff_qf = 1.0; }

   // The transformation matrix has to be "mixed" value and integral map type,
   // which means that the coefficient has to be scaled like the Jacobian
   // determinant.
   if (convert_map_type)
   {
      const int n = W_mix_coeff_qf.Size();
      const real_t *d_detJ = geom->detJ.Read();
      real_t *d_w_mix = W_mix_coeff_qf.ReadWrite();
      real_t *d_w = W_coeff_qf.ReadWrite();
      const bool zero_l2 = zero_l2_block;
      MFEM_FORALL(i, n,
      {
         const real_t detJ = d_detJ[i];
         if (!zero_l2) { d_w[i] *= detJ*detJ; }
         d_w_mix[i] *= detJ;
      });
   }

   L_inv.reset(new DGMassInverse(fes_l2, W_mix_coeff));

   if (zero_l2_block)
   {
      A_11.reset();
   }
   else
   {
      mass_l2.Assemble();
      mass_l2.AssembleDiagonal(L_diag);
      mass_l2.FormSystemMatrix(empty, L);

      A_11.reset(new RAPOperator(*L_inv, *L, *L_inv));

      if (mode == GRAD_DIV)
      {
         L_diag_unweighted.SetSize(L_diag.Size());

         BilinearForm mass_l2_mix(&fes_l2);
         mass_l2_mix.AddDomainIntegrator(new MassIntegrator(W_mix_coeff));
         mass_l2_mix.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         mass_l2_mix.Assemble();
         mass_l2_mix.AssembleDiagonal(L_diag_unweighted);
      }

      const real_t *d_L_diag_unweighted = L_diag_unweighted.Read();
      real_t *d_L_diag = L_diag.ReadWrite();
      MFEM_FORALL(i, L_diag.Size(),
      {
         const real_t d = d_L_diag_unweighted[i];
         d_L_diag[i] /= d*d;
      });
   }

   // Reassemble the RT mass operator with the new coefficient
   mass_rt.Update();
   mass_rt.Assemble();
   mass_rt.FormSystemMatrix(ess_rt_dofs, R);

   // Form the updated approximate Schur complement
   mass_rt.AssembleDiagonal(R_diag);

   // Update the mass RT diagonal for essential DOFs
   {
      const int *d_I = ess_rt_dofs.Read();
      real_t *d_R_diag = R_diag.ReadWrite();
      MFEM_FORALL(i, ess_rt_dofs.Size(), d_R_diag[d_I[i]] = 1.0;);
   }

   // Form the approximate Schur complement
   {
      Reciprocal(R_diag);
      std::unique_ptr<HypreParMatrix> R_diag_inv(MakeDiagonalMatrix(R_diag, fes_rt));
      if (zero_l2_block)
      {
         S.reset(RAP(R_diag_inv.get(), Dt.get()));
      }
      else
      {
         std::unique_ptr<HypreParMatrix> D_Minv_Dt(RAP(R_diag_inv.get(), Dt.get()));
         std::unique_ptr<HypreParMatrix> L_diag_inv(MakeDiagonalMatrix(L_diag, fes_l2));
         S.reset(ParAdd(D_Minv_Dt.get(), L_diag_inv.get()));
      }
   }

   // Reassemble the preconditioners
   R_inv.reset(new OperatorJacobiSmoother(mass_rt, ess_rt_dofs));
   S_inv.SetOperator(*S);

   // Set up the block operators
   A_block.reset(new BlockOperator(offsets));
   // Omit the (1,1)-block when the L coefficient is identically zero.
   if (A_11) { A_block->SetBlock(0, 0, A_11.get()); }
   A_block->SetBlock(0, 1, D.get());
   A_block->SetBlock(1, 0, Dt.get());
   A_block->SetBlock(1, 1, R.Ptr(), -1.0);

   D_prec.reset(new BlockDiagonalPreconditioner(offsets));
   D_prec->SetDiagonalBlock(0, &S_inv);
   D_prec->SetDiagonalBlock(1, R_inv.get());

   minres.SetPreconditioner(*D_prec);
   minres.SetOperator(*A_block);
}

void HdivSaddlePointSolver::EliminateBC(Vector &b) const
{
   const int n_ess_dofs = ess_rt_dofs.Size();
   if (fes_l2.GetParMesh()->ReduceInt(n_ess_dofs) == 0) { return; }

   const int n_l2 = offsets[1];
   const int n_rt = offsets[2]-offsets[1];
   Vector bE(b, 0, n_l2);
   Vector bF(b, n_l2, n_rt);

   // SetBC must be called first
   MFEM_VERIFY(x_bc.Size() == n_rt || n_ess_dofs == 0, "BCs not set");

   // Create a vector z that has the BC values at essential DOFs, zero elsewhere
   z.SetSize(n_rt);
   z.UseDevice(true);
   z = 0.0;
   const int *d_I = ess_rt_dofs.Read();
   const real_t *d_x_bc = x_bc.Read();
   real_t *d_z = z.ReadWrite();
   MFEM_FORALL(i, n_ess_dofs,
   {
      const int j = d_I[i];
      d_z[j] = d_x_bc[j];
   });

   // Convert to the IntegratedGLL basis used internally
   w.SetSize(n_rt);
   basis_rt.MultInverse(z, w);

   // Eliminate the BCs in the L2 RHS
   D_e->Mult(-1.0, w, 1.0, bE);

   // Eliminate the BCs in the RT RHS
   // Flip the sign because the R block appears with multiplier -1
   z.SetSize(n_rt);
   R_e->Mult(w, z);
   bF += z;

   // Insert the RT BCs into the RHS at the essential DOFs.
   const real_t *d_w = w.Read();
   real_t *d_bF = bF.ReadWrite(); // Need read-write access to set subvector
   MFEM_FORALL(i, n_ess_dofs,
   {
      const int j = d_I[i];
      d_bF[j] = -d_w[j];
   });

   // Make sure the monolithic RHS is updated
   bE.SyncAliasMemory(b);
   bF.SyncAliasMemory(b);
}

void HdivSaddlePointSolver::Mult(const Vector &b, Vector &x) const
{
   w.SetSize(fes_l2.GetTrueVSize());
   b_prime.SetSize(b.Size());
   x_prime.SetSize(x.Size());

   // Transform RHS to the IntegratedGLL basis
   Vector bE_prime(b_prime, offsets[0], offsets[1]-offsets[0]);
   Vector bF_prime(b_prime, offsets[1], offsets[2]-offsets[1]);

   const Vector bE(const_cast<Vector&>(b), offsets[0], offsets[1]-offsets[0]);
   const Vector bF(const_cast<Vector&>(b), offsets[1], offsets[2]-offsets[1]);

   z.SetSize(bE.Size());
   basis_l2.MultTranspose(bE, z);
   basis_rt.MultTranspose(bF, bF_prime);
   // Transform by the inverse of the L2 mass matrix
   L_inv->Mult(z, bE_prime);

   // Update the monolithic transformed RHS
   bE_prime.SyncAliasMemory(b_prime);
   bF_prime.SyncAliasMemory(b_prime);

   // Eliminate the RT essential BCs
   EliminateBC(b_prime);

   // Solve the transformed system
   minres.Mult(b_prime, x_prime);

   // Transform the solution back to the user's basis
   Vector xE_prime(x_prime, offsets[0], offsets[1]-offsets[0]);
   Vector xF_prime(x_prime, offsets[1], offsets[2]-offsets[1]);

   Vector xE(x, offsets[0], offsets[1]-offsets[0]);
   Vector xF(x, offsets[1], offsets[2]-offsets[1]);

   z.SetSize(bE.Size()); // Size of z may have changed in EliminateBC
   L_inv->Mult(xE_prime, z);

   basis_l2.Mult(z, xE);
   basis_rt.Mult(xF_prime, xF);

   // Update the monolithic solution vector
   xE.SyncAliasMemory(x);
   xF.SyncAliasMemory(x);
}

} // namespace mfem

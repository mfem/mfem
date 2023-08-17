// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "darcy_solver.hpp"

using namespace std;
// using namespace mfem;
// using namespace blocksolvers;

namespace mfem
{
namespace blocksolvers
{

/// Exact solutions 
void u_exact(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double zi(x.Size() == 3 ? x(2) : 0.0);

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);
   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

double p_exact(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(x.Size() == 3 ? x(2) : 0.0);
   return exp(xi)*sin(yi)*cos(zi);
}

void f_exact(const Vector & x, Vector & f)
{
   f = 0.0;
}

double g_exact(const Vector & x)
{
   if (x.Size() == 3) { return -p_exact(x); }
   return 0;
}

double natural_bc(const Vector & x)
{
   return (-p_exact(x));
}

/// Check if using Neumann BC
bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr)
{
   for (int attr : ess_bdr_attr) { if (attr == 0) { return false; } }
   return true;
}

/// Set standard options for IterativeSolvers
void SetOptions(IterativeSolver& solver, const IterSolveParameters& param)
{
   solver.SetPrintLevel(param.print_level);
   solver.SetMaxIter(param.max_iter);
   solver.SetAbsTol(param.abs_tol);
   solver.SetRelTol(param.rel_tol);
}

SparseMatrix ElemToDof(const ParFiniteElementSpace& fes)
{
   int* I = new int[fes.GetNE()+1];
   copy_n(fes.GetElementToDofTable().GetI(), fes.GetNE()+1, I);
   Array<int> J(new int[I[fes.GetNE()]], I[fes.GetNE()]);
   copy_n(fes.GetElementToDofTable().GetJ(), J.Size(), J.begin());
   fes.AdjustVDofs(J);
   double* D = new double[J.Size()];
   fill_n(D, J.Size(), 1.0);
   return SparseMatrix(I, J, D, fes.GetNE(), fes.GetVSize());
}

DFSSpaces::DFSSpaces(int order, int num_refine, ParMesh *mesh,
                     const Array<int>& ess_attr, const DFSParameters& param)
   : hdiv_fec_(order, mesh->Dimension()), l2_fec_(order, mesh->Dimension()),
     l2_0_fec_(0, mesh->Dimension()), ess_bdr_attr_(ess_attr), level_(0)
{
   if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order)
   {
      mfem_error("DFSDataCollector: High order spaces on tetrahedra are not supported");
   }

   data_.param = param;

   if (mesh->Dimension() == 3)
   {
      hcurl_fec_.reset(new ND_FECollection(order+1, mesh->Dimension()));
   }
   else
   {
      hcurl_fec_.reset(new H1_FECollection(order+1, mesh->Dimension()));
   }

   all_bdr_attr_.SetSize(ess_attr.Size(), 1);
   hdiv_fes_.reset(new ParFiniteElementSpace(mesh, &hdiv_fec_));
   l2_fes_.reset(new ParFiniteElementSpace(mesh, &l2_fec_));
   coarse_hdiv_fes_.reset(new ParFiniteElementSpace(*hdiv_fes_));
   coarse_l2_fes_.reset(new ParFiniteElementSpace(*l2_fes_));
   l2_0_fes_.reset(new ParFiniteElementSpace(mesh, &l2_0_fec_));
   l2_0_fes_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
   el_l2dof_.reserve(num_refine+1);
   el_l2dof_.push_back(ElemToDof(*coarse_l2_fes_));

   data_.agg_hdivdof.resize(num_refine);
   data_.agg_l2dof.resize(num_refine);
   data_.P_hdiv.resize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
   data_.P_l2.resize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
   data_.Q_l2.resize(num_refine);
   hdiv_fes_->GetEssentialTrueDofs(ess_attr, data_.coarsest_ess_hdivdofs);
   data_.C.resize(num_refine+1);

   hcurl_fes_.reset(new ParFiniteElementSpace(mesh, hcurl_fec_.get()));
   coarse_hcurl_fes_.reset(new ParFiniteElementSpace(*hcurl_fes_));
   data_.P_hcurl.resize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
}

// Darcy problem function
DarcyProblem::DarcyProblem(Mesh &mesh, int num_refs, int order,
                           const char *coef_file, Array<int> &ess_bdr,
                           DFSParameters dfs_param)
   : mesh_(MPI_COMM_WORLD, mesh), ucoeff_(mesh.Dimension(), u_exact),
     pcoeff_(p_exact), dfs_spaces_(order, num_refs, &mesh_, ess_bdr, dfs_param),
     mass_coeff()
{
   for (int l = 0; l < num_refs; l++)
   {
      mesh_.UniformRefinement();
      dfs_spaces_.CollectDFSData();
   }

   Vector coef_vector(mesh.GetNE());
   coef_vector = 1.0;
   if (std::strcmp(coef_file, ""))
   {
      ifstream coef_str(coef_file);
      coef_vector.Load(coef_str, mesh.GetNE());
   }

   mass_coeff.UpdateConstants(coef_vector);
   VectorFunctionCoefficient fcoeff(mesh_.Dimension(), f_exact);
   FunctionCoefficient natcoeff(natural_bc);
   FunctionCoefficient gcoeff(g_exact);

   u_.SetSpace(dfs_spaces_.GetHdivFES());
   p_.SetSpace(dfs_spaces_.GetL2FES());
   p_ = 0.0;
   u_ = 0.0;
   u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

   ParLinearForm fform(dfs_spaces_.GetHdivFES());
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();

   ParLinearForm gform(dfs_spaces_.GetL2FES());
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();

   // ParBilinearForm mVarf_(dfs_spaces_.GetHdivFES());
   // ParMixedBilinearForm bVarf_(dfs_spaces_.GetHdivFES(), dfs_spaces_.GetL2FES());

   mVarf_ = make_shared<ParBilinearForm>(dfs_spaces_.GetHdivFES());
   bVarf_ = make_shared<ParMixedBilinearForm>(dfs_spaces_.GetHdivFES(),
                                              dfs_spaces_.GetL2FES());

   mVarf_->AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   mVarf_->ComputeElementMatrices();
   mVarf_->Assemble();
   mVarf_->EliminateEssentialBC(ess_bdr, u_, fform);
   mVarf_->Finalize();
   M_.Reset(mVarf_->ParallelAssemble());

   bVarf_->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf_->Assemble();
   bVarf_->SpMat() *= -1.0;
   bVarf_->EliminateTrialDofs(ess_bdr, u_, gform);
   bVarf_->Finalize();
   B_.Reset(bVarf_->ParallelAssemble());

   rhs_.SetSize(M_->NumRows() + B_->NumRows());
   Vector rhs_block0(rhs_.GetData(), M_->NumRows());
   Vector rhs_block1(rhs_.GetData()+M_->NumRows(), B_->NumRows());
   fform.ParallelAssemble(rhs_block0);
   gform.ParallelAssemble(rhs_block1);

   ess_data_.SetSize(M_->NumRows() + B_->NumRows());
   ess_data_ = 0.0;
   Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
   u_.ParallelProject(ess_data_block0);

   int order_quad = max(2, 2*order+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs_[i] = &(IntRules.Get(i, order_quad));
   }
}

void DarcyProblem::ShowError(const Vector& sol, bool verbose)
{
   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
   double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
   double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

   if (!verbose) { return; }
   cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

void DarcyProblem::VisualizeSolution(const Vector& sol, string tag)
{
   int num_procs, myid;
   MPI_Comm_size(mesh_.GetComm(), &num_procs);
   MPI_Comm_rank(mesh_.GetComm(), &myid);

   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   const char vishost[] = "localhost";
   const int  visport   = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock.precision(8);
   u_sock << "solution\n" << mesh_ << u_ << "window_title 'Velocity ("
          << tag << " solver)'" << endl;
   MPI_Barrier(mesh_.GetComm());
   socketstream p_sock(vishost, visport);
   p_sock << "parallel " << num_procs << " " << myid << "\n";
   p_sock.precision(8);
   p_sock << "solution\n" << mesh_ << p_ << "window_title 'Pressure ("
          << tag << " solver)'" << endl;
}

/// Wrapper Block Diagonal Preconditioned MINRES (ex5p)
BDPMinresSolver::BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                                 IterSolveParameters param)
   : DarcySolver(M.NumRows(), B.NumRows()), op_(offsets_), prec_(offsets_),
     BT_(B.Transpose()), solver_(M.GetComm())
{
   op_.SetBlock(0,0, &M);
   op_.SetBlock(0,1, BT_.As<HypreParMatrix>());
   op_.SetBlock(1,0, &B);

   Vector Md;
   M.GetDiag(Md);
   BT_.As<HypreParMatrix>()->InvScaleRows(Md);
   S_.Reset(ParMult(&B, BT_.As<HypreParMatrix>()));
   BT_.As<HypreParMatrix>()->ScaleRows(Md);

   prec_.SetDiagonalBlock(0, new HypreDiagScale(M));
   prec_.SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
   static_cast<HypreBoomerAMG&>(prec_.GetDiagonalBlock(1)).SetPrintLevel(0);
   prec_.owns_blocks = true;

   SetOptions(solver_, param);
   solver_.SetOperator(op_);
   solver_.SetPreconditioner(prec_);
}

void BDPMinresSolver::Mult(const Vector & x, Vector & y) const
{
   solver_.Mult(x, y);
   for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
}

} // namespace blocksolvers
} // namespace mfem

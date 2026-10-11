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

#include "solver_utils.hpp"

#ifdef MFEM_USE_MUMPS
#include "linalg/mumps.hpp"
#endif

#ifdef MFEM_USE_MKL_CPARDISO
#include "linalg/cpardiso.hpp"
#endif

#ifdef MFEM_USE_SUPERLU
#include "linalg/superlu.hpp"
#endif

#ifdef MFEM_USE_STRUMPACK
#include "linalg/strumpack.hpp"
#endif


namespace mfem
{

static ParallelDirectSolver::Type ParseType(const std::string &name_in)
{
   std::string name = name_in;
   for (char &c : name) { c = std::tolower(c); }

   if (name == "mumps")      { return ParallelDirectSolver::Type::MUMPS; }
   if (name == "superlu")    { return ParallelDirectSolver::Type::SUPERLU; }
   if (name == "strumpack")  { return ParallelDirectSolver::Type::STRUMPACK; }
   if (name == "cpardiso")   { return ParallelDirectSolver::Type::CPARDISO; }
   if (name == "auto")       { return ParallelDirectSolver::Type::AUTO; }

   MFEM_ABORT("Unknown ParallelDirectSolver type string: " + name_in);
   return ParallelDirectSolver::Type::AUTO; // unreachable
}

ParallelDirectSolver::ParallelDirectSolver(MPI_Comm comm_, Type type_)
   : type(type_), comm(comm_)
{
   if (type == Type::AUTO)
   {
#ifdef MFEM_USE_MUMPS
      type = Type::MUMPS;
#elif defined(MFEM_USE_SUPERLU)
      type = Type::SUPERLU;
#elif defined(MFEM_USE_STRUMPACK)
      type = Type::STRUMPACK;
#elif defined(MFEM_USE_MKL_CPARDISO)
      type = Type::CPARDISO;
#else
      MFEM_ABORT("No parallel direct solver was enabled in MFEM.");
#endif
   }

   InitSolver();
}

ParallelDirectSolver::ParallelDirectSolver(MPI_Comm comm,
                                           const std::string &name)
   : ParallelDirectSolver(comm, ParseType(name)) { }

void ParallelDirectSolver::InitSolver()
{
   switch (type)
   {
      case Type::MUMPS:
#ifdef MFEM_USE_MUMPS
      {
         auto *mumps = new MUMPSSolver(comm);
         solver.reset(mumps);
      }
      break;
#else
      MFEM_ABORT("MUMPS requested but MFEM_USE_MUMPS is not defined.");
#endif

      case Type::SUPERLU:
#ifdef MFEM_USE_SUPERLU
      {
         auto *slu = new SuperLUSolver(comm);
         solver.reset(slu);
      }
      break;
#else
      MFEM_ABORT("SuperLU requested but MFEM_USE_SUPERLU is not defined.");
#endif

      case Type::STRUMPACK:
#ifdef MFEM_USE_STRUMPACK
      {
         auto *strumpack = new STRUMPACKSolver(comm);
         strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
         strumpack->SetMatching(strumpack::MatchingJob::NONE);
         strumpack->SetCompression(strumpack::CompressionType::NONE);
         solver.reset(strumpack);
      }
      break;
#else
      MFEM_ABORT("STRUMPACK requested but MFEM_USE_STRUMPACK is not defined.");
#endif

      case Type::CPARDISO:
#ifdef MFEM_USE_MKL_CPARDISO
      {
         auto *cpardiso = new CPardisoSolver(comm);
         solver.reset(cpardiso);
      }
      break;
#else
      MFEM_ABORT("CPARDISO requested but MFEM_USE_MKL_CPARDISO is not defined.");
#endif

      default:
         MFEM_ABORT("Invalid solver type.");
   }
}

void ParallelDirectSolver::SetOperator(const Operator &op)
{
   MFEM_VERIFY(solver, "Solver not initialized.");

   switch (type)
   {
      case Type::MUMPS:
      case Type::CPARDISO:
         // These accept HypreParMatrix directly (or Operator that dynamic_casts to it)
         solver->SetOperator(op);
         break;

      case Type::SUPERLU:
#ifdef MFEM_USE_SUPERLU
         // SuperLUSolver requires a SuperLURowLocMatrix.
         superlu_mat.reset(new SuperLURowLocMatrix(op));
         solver->SetOperator(*superlu_mat);
         break;
#else
         MFEM_ABORT("SUPERLU not enabled.");
#endif

      case Type::STRUMPACK:
#ifdef MFEM_USE_STRUMPACK
         // STRUMPACKSolver requires a STRUMPACKRowLocMatrix.
         strumpack_mat.reset(new STRUMPACKRowLocMatrix(op));
         solver->SetOperator(*strumpack_mat);
         break;
#else
         MFEM_ABORT("STRUMPACK not enabled.");
#endif

      default:
         MFEM_ABORT("SetOperator: unknown type.");
   }
}

void ParallelDirectSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(solver, "Solver not initialized.");
   solver->Mult(x, y);
}

void ParallelDirectSolver::SetPrintLevel(int print_lvl)
{
   if (!solver) { return; }

#ifdef MFEM_USE_MUMPS
   if (auto *mumps = dynamic_cast<MUMPSSolver*>(solver.get()))
   {
      mumps->SetPrintLevel(print_lvl);
      return;
   }
#endif
#ifdef MFEM_USE_SUPERLU
   if (auto *slu = dynamic_cast<SuperLUSolver*>(solver.get()))
   {
      slu->SetPrintStatistics(print_lvl != 0);
      return;
   }
#endif
#ifdef MFEM_USE_STRUMPACK
   if (auto *sp = dynamic_cast<STRUMPACKSolver*>(solver.get()))
   {
      sp->SetPrintFactorStatistics(print_lvl != 0);
      sp->SetPrintSolveStatistics(print_lvl != 0);
      return;
   }
#endif
#ifdef MFEM_USE_MKL_CPARDISO
   if (auto *pardiso = dynamic_cast<CPardisoSolver*>(solver.get()))
   {
      pardiso->SetPrintLevel(print_lvl);
      return;
   }
#endif
}

AMGFSchwarzSolver::AMGFSchwarzSolver(const HypreParMatrix &J,
                                     const HypreParMatrix &P,
                                     std::function<void(Vector &)> get_D_,
                                     real_t D_threshold_)
   : AMGFSolver(), comm(J.GetComm()), get_D(std::move(get_D_)),
     D_threshold(D_threshold_)
{
   SetFilteredSubspaceTransferOperator(P);

   // Row i of J P holds the nonzeros of row i of J whose dofs are in the
   // filtered subspace, with columns numbered by subspace dof.
   std::unique_ptr<HypreParMatrix> JP(ParMult(&J, &P));
   JP->HostRead();
   SparseMatrix diag, offd;
   HYPRE_BigInt *cmap;
   JP->GetDiag(diag);
   JP->GetOffd(offd, cmap);
   const HYPRE_BigInt first_col =
      hypre_ParCSRMatrixFirstColDiag(static_cast<hypre_ParCSRMatrix *>(*JP));
   const int *diag_I = diag.HostReadI(), *diag_J = diag.HostReadJ();
   const int *offd_I = offd.HostReadI(), *offd_J = offd.HostReadJ();
   const real_t *diag_V = diag.HostReadData(), *offd_V = offd.HostReadData();

   row_patches.resize(JP->Height());
   for (int i = 0; i < JP->Height(); i++)
   {
      for (int k = diag_I[i]; k < diag_I[i+1]; k++)
      {
         if (diag_V[k] != 0.0) { row_patches[i].Append(first_col + diag_J[k]); }
      }
      for (int k = offd_I[i]; k < offd_I[i+1]; k++)
      {
         if (offd_V[k] != 0.0) { row_patches[i].Append(cmap[offd_J[k]]); }
      }
   }
}

void AMGFSchwarzSolver::SetOperator(const Operator &op)
{
   AMGFSolver::SetOperator(op);

   Vector D;
   get_D(D);
   MFEM_VERIFY(D.Size() >= static_cast<int>(row_patches.size()),
               "AMGFSchwarzSolver: D has fewer entries than the rows of J.");
   const real_t *h_D = D.HostRead();

   std::vector<Array<HYPRE_BigInt>> patches;
   for (size_t i = 0; i < row_patches.size(); i++)
   {
      if (row_patches[i].Size() > 0 && h_D[i] >= D_threshold)
      {
         patches.push_back(row_patches[i]);
      }
   }
   HYPRE_BigInt num_local = patches.size(), num_global;
   MPI_Allreduce(&num_local, &num_global, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   num_patches.Append(num_global);

   // Replace the subspace solver instead of updating it, so that it is only
   // set up once, when FilteredSolver hands it the new subspace operator.
   schwarz.reset(new AdditiveSchwarz);
   schwarz->SetSubdomains(patches);
   SetFilteredSubspaceSolver(*schwarz);
}

}

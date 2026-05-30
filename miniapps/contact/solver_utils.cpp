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

}

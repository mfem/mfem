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

#ifndef MFEM_FILTEREDSOLVER
#define MFEM_FILTEREDSOLVER

#include "../config/config.hpp"
#include "operator.hpp"


namespace mfem
{

/**
 * @brief FilteredSolver:
 *
 * Interface for the FilteredSolver solver
 */
class FilteredSolver : public Solver
{
public:
   /**
    * @brief Constructor with MPI_Comm parameter.
    */
   FilteredSolver(MPI_Comm comm_);

   /**
    * @brief Constructor with a generic Operator.
    */
   FilteredSolver(const Operator &op);

   /**
    * @brief Set the Operator and perform factorization
    */
   void SetOperator(const Operator &op);

   void Mult(const Vector &x, Vector &y) const;

   /**
    * @brief Set the error print level
    *
    * Supported values are:
    * - 0:  No output printed
    * - 1:  Only errors printed
    * - 2:  Errors, warnings, and main stats printed
    * - 3:  Errors, warning, main stats, and terse diagnostics printed
    * - 4:  Errors, warning, main stats, diagnostics, and input/output printed
    *
    * @param print_lvl Print level, default is 2
    *
    * @note This method has to be called before SetOperator
    */
   void SetPrintLevel(int print_lvl);

   // Destructor
   ~FilteredSolver();

private:
   // MPI communicator
   MPI_Comm comm;

   // Number of procs
   int numProcs;

   // MPI rank
   int myid;

   // Parameter controlling the printing level
   int print_level;

   /// Method for initialization
   void Init(MPI_Comm comm_);

}; // mfem::FilteredSolver class

} // namespace mfem

#endif

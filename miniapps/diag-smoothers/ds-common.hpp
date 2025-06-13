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

#ifndef MFEM_DS_COMMON_HPP
#define MFEM_DS_COMMON_HPP

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

namespace ds_common
{

extern int MONITOR_DIGITS;
extern int MG_MAX_ITER;
extern real_t MG_REL_TOL;

extern int dim;
extern int space_dim;
extern real_t freq;
extern real_t kappa;

// Enumerator for the different solvers to implement
enum SolverType
{
   sli,
   cg,
   num_solvers,  // last
};

// Enumerator for the different integrators to implement
enum IntegratorType
{
   mass,
   diffusion,
   maxwell,
   num_integrators,  // last
};

// Enumerator for the different types of preconditioners
enum PCType
{
   none,
   abs_global,
   pq_element,
   num_pc,  // last
};

/// @brief Custom monitor that prints a csv-formatted file
class DataMonitor : public IterativeSolverMonitor
{
private:
   ofstream os;
   int precision;
public:
   DataMonitor(string file_name, int ndigits);
   void MonitorResidual(int it, real_t norm, const Vector &x, bool final);
   void MonitorSolution(int it, real_t norm, const Vector &x, bool final);
};


/// @brief Abs-L(1)-Jacobi custom general geometric multigrid method.
///
/// Intermediate levels use Abs-L(1)-Jacobi preconditioner by applying the
/// level matrix to the constant vector one. These are wrapped by an
/// OperatorJacobiSmoother. Coarsest level uses a used-selected solver
/// with an Abs-L(1)-Jacobi smoother. The assembly level is user-defined.
///
/// @warning The construction of the smoother is based on the application of
/// AbsMult, which usually unfolds component-wise. E.g., if `A = B C`, then
/// `|A|x = |B|(|C| x)`.
class AbsL1GeometricMultigrid : public GeometricMultigrid
{
public:
   AbsL1GeometricMultigrid(ParFiniteElementSpaceHierarchy& fes_hierarchy,
                           Array<int>& ess_bdr,
                           IntegratorType it,
                           SolverType st,
                           AssemblyLevel al);

   ~AbsL1GeometricMultigrid() { delete coarse_pc; }

   bool GetOwnershipLevelOperators() const { return mg_owned; }

private:
   IntegratorType integrator_type;
   SolverType solver_type;
   AssemblyLevel assembly_level;
   bool mg_owned;
   OperatorJacobiSmoother* coarse_pc;
   ConstantCoefficient one;

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace);

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace, int level);

   void ConstructBilinearForm(ParFiniteElementSpace& fespace);

};

void AssembleElementLpqJacobiDiag(ParBilinearForm& form, real_t p, real_t q,
                                  Vector& diag);

real_t diffusion_solution(const Vector &x);
real_t diffusion_source(const Vector &x);

void maxwell_solution(const Vector &x, Vector &u);
void maxwell_source(const Vector &x, Vector &f);

} // namespace ds_common

#endif // MFEM_DS_COMMON_HPP

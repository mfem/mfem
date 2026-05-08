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

#ifndef MFEM_NAVIER_TOOLS_HPP
#define MFEM_NAVIER_TOOLS_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_GSLIB

#include <vector>
#include <type_traits>

namespace mfem
{
namespace navier
{
   void ComputeCurl3D_particle(ParGridFunction &u, ParGridFunction &cu);

   /**
 * @brief Class for solving Poisson's equation:
 *
 *       - ∇ ⋅(κ ∇ u) = f  in Ω
 *
 */
class DiffusionSolver_particle
{
private:
   Mesh * mesh = nullptr;
   int order = 1;
   // diffusion coefficient
   Coefficient * diffcf = nullptr;
   // mass coefficient
   Coefficient * masscf = nullptr;
   Coefficient * rhscf = nullptr;
   Coefficient * essbdr_cf = nullptr;
   Coefficient * neumann_cf = nullptr;
   VectorCoefficient * gradient_cf = nullptr;

   // FEM solver
   int dim;
   FiniteElementCollection * fec = nullptr;
   FiniteElementSpace * fes = nullptr;
   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   Array<int> neumann_bdr;
   GridFunction * u = nullptr;
   LinearForm * b = nullptr;
   BilinearForm * a = nullptr;
   OperatorPtr A;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   DiffusionSolver_particle() { }
   DiffusionSolver_particle(Mesh * mesh_, int order_, Coefficient * diffcf_,
                   Coefficient * cf_);

   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetOrder(int order_) { order = order_ ; }
   void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
   void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
   void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
   void SetEssentialBoundary(const Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
   void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
   void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

   void ResetFEM();
   void SetupFEM();

   void UpdateEssentialTDofs();
   void AssembleDiffusionBilinear(bool update_ess_tdofs=true);
   void Solve();
   GridFunction * GetFEMSolution();
   LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   ParGridFunction * GetParFEMSolution();
   ParLinearForm * GetParLinearForm()
   {
      if (parallel)
      {
         return dynamic_cast<ParLinearForm *>(b);
      }
      else
      {
         MFEM_ABORT("Wrong code path. Call GetLinearForm");
         return nullptr;
      }
   }
#endif

   ~DiffusionSolver_particle();
};

} // namespace navier

} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_NAVIER_PARTICLES_HPP

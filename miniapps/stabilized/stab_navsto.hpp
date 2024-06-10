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

#ifndef MFEM_STAB_NAVSTO_HPP
#define MFEM_STAB_NAVSTO_HPP

#include "mfem.hpp"
#include "stab_tau.hpp"

namespace mfem
{

/** Stabilized incompressible Navier-Stokes integrator
    Start with Galerkin for stokes - done
    Add convection - done
    Modify diffusion - done
    Add difussion to residual - done
    CHECK NUMBERING HESSIAN -> NURBS = WRONG?? 2D = ok --> 3D???   --> DONE NEEDS CHECKING???
    Inverse estimate check order -> done
    Add force                    -> done

    Add supg   -  rhs done, jac conv + press --> ignore diffusion for now
    Add pspg   -  rhs done, jac conv + press --> ignore diffusion for now
    Add lsic    -  rhs done, jac conv + press --> ignore diffusion for now

    Add correct inverse estimate -> done?? number does not coincide with H&C

    Add VMS/GLS
    Add selection option of different stab modes

    Parallel --> almost their??

    Leopoldo P. Franca, Sérgio L. Frey
    Stabilized finite element methods:
    II. The incompressible Navier-Stokes equations.
    Computer Methods in Applied Mechanics and Engineering, 99(2-3), 209-233.

    https://doi.org/10.1016/0045-7825(92)90041-H
    https://www.sciencedirect.com/science/article/pii/004578259290041H

*/
class StabInNavStoIntegrator : public BlockNonlinearFormIntegrator
{
private:
   Coefficient *c_mu;
   VectorCoefficient *c_force;
   Vector u, f, grad_p;
   DenseMatrix flux;

   DenseMatrix elf_u, elv_u;
//   Vector elf_u, elv_u;//
   DenseMatrix elf_p, elv_p;
   Vector sh_u, ushg_u, sh_p;
   DenseMatrix shg_u, shh_u, shg_p, grad_u, hess_u;
   Array2D<int> hmap;

   /// The stabilization parameters
   StabType stab;
   Tau *tau = nullptr;
   Tau *delta = nullptr;
   Vector res, up;

   /// The advection field
   VectorCoefficient *adv = nullptr; // tbd???

   int dim = -1;
   void SetDim(int dim);

public:
   StabInNavStoIntegrator(Coefficient &mu_,
                          VectorCoefficient &force_,
                          Tau &t, Tau &d,
                          StabType s = GALERKIN);

   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *> &elfun);

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats);
};

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
   GeneralResidualMonitor(const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
      print_level = print_lvl;
   }

   GeneralResidualMonitor(MPI_Comm comm,
                          const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
   }

   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final);

private:
   const std::string prefix;
   int print_level;
   mutable real_t norm0;
};

class SystemResidualMonitor : public IterativeSolverMonitor
{
public:
   SystemResidualMonitor(const std::string& prefix_,
                         int print_lvl,
                         Array<int> &offsets,
                         DataCollection *dc_ = nullptr)
      : prefix(prefix_), bOffsets(offsets), dc(dc_)
   {
      print_level = print_lvl;
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
   }

   SystemResidualMonitor(MPI_Comm comm,
                         const std::string& prefix_,
                         int print_lvl,
                         Array<int> &offsets,
                         DataCollection *dc_ = nullptr)
      : prefix(prefix_), bOffsets(offsets), dc(dc_)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
   }


   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final);

private:
   const std::string prefix;
   int print_level, nvar;
   mutable Vector norm0;
  // Offsets for extracting block vector segments
   Array<int> &bOffsets;
   DataCollection *dc;
};

// Custom block preconditioner for the Jacobian
class JacobianPreconditioner : public BlockLowerTriangularPreconditioner //BlockDiagonalPreconditioner
{
protected:
   Array<Solver *> prec;
public:
   JacobianPreconditioner(Array<int> &offsets, Array<Solver *> p)
     : BlockLowerTriangularPreconditioner (offsets), prec(p)
   { MFEM_VERIFY(offsets.Size()-1 == p.Size(), ""); };

   virtual void SetOperator(const Operator &op);

   virtual ~JacobianPreconditioner();
};


} // namespace mfem

#endif

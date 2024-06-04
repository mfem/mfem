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

namespace mfem
{

/** Stabilized incompressible Navier-Stokes integrator
   Start with Galerkin for stokes

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
   DenseMatrix elf_u,elv_u;
   DenseMatrix elf_p,elv_p;
   Vector sh_u,sh_p;
   DenseMatrix shg_u, grad_u;
   Vector u;
   DenseMatrix sigma;

public:
   StabInNavStoIntegrator(Coefficient &mu_) : c_mu(&mu_) { }

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

} // namespace mfem

#endif

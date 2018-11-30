// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM4_INTEGRATORS_DIFFUSION_HPP
#define MFEM4_INTEGRATORS_DIFFUSION_HPP

#include "fem/bilininteg.hpp"

namespace mfem4
{

using namespace mfem;


struct ElementGeometries
{
   Tensor nodes;
   Tensor J, invJ, detJ;

   int GetNE() const;
};


struct ElementAssembly
{
   Tensor dof_quad, quad_dof;
   Tensor quad_data;

   int GetNE() const;
};


/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v) where Q
    can be a scalar or a matrix coefficient. */
class DiffusionIntegrator: public BilinearFormIntegrator
{
public:
   /// Construct a diffusion integrator with coefficient Q = 1
   DiffusionIntegrator() : Q(NULL), MQ(NULL) {}

   /// Construct a diffusion integrator with a scalar coefficient q
   DiffusionIntegrator(Coefficient &q) : Q(&q), MQ(NULL) {}

   /// Construct a diffusion integrator with a matrix coefficient q
   DiffusionIntegrator(MatrixCoefficient &q) : Q(NULL), MQ(&q) {}


   /** Given a batch of elements, the trial and test FiniteElements,
       compute the element stiffness matrices. */
   virtual void AssembleElements(const Array<int> &batch,
                                 const FiniteElement &trial_fe,
                                 const FiniteElement &test_fe,
                                 const ElementGeometries &geoms,
                                 DenseTensor &matrices);

   /** Given a batch of elements, the trial and test FiniteElements,
       partial assemble dof<->quad maps, and geometry and coefficient data
       at quadrature points. */
   virtual void PartialAssemble(const Array<int> &batch,
                                const FiniteElement &trial_fe,
                                const FiniteElement &test_fe,
                                const ElementGeometries &geoms,
                                ElementAssembly &easm);

   /** Given partially assembled data from PartialAssemble, apply
       element-wise operators on vector @a x and return @a y. */
   virtual void MultAdd(const ElementAssembly &easm,
                        const Vector &x, Vector &y);


   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, int with_coef = 1);

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);

protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
};


} // namespace mfem4

#endif // MFEM4_INTEGRATORS_DIFFUSION_HPP

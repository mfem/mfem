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
#include "mesh/batchgeom.hpp"

namespace mfem4
{

using namespace mfem;

/** Base class for a (partially) assembled batch of elements. The base class
 *  is opaque and can only destruct itself. The format of the data and the
 *  exact partial assembly algorithm depends on each integrator.
 */
struct AssemblyData
{
   virtual ~AssemblyData() {}
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
   virtual void AssembleElements(const BatchGeometry &geom,
                                 const FiniteElement &trial_fe,
                                 const FiniteElement &test_fe,
                                 DenseTensor &matrices);

   /** Given a batch of elements, the trial and test FiniteElements,
       perform partial assembly and return opaque data that can be passed
       to MultAdd to perform the action of the operator. */
   virtual AssemblyData* PartialAssemble(const BatchGeometry &geom,
                                         const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe);

   /** Given partially assembled data from PartialAssemble, apply
       element-wise operators on vector @a x and return @a y. */
   virtual void MultAdd(const AssemblyData &assembly,
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

   /// DiffusionIntegrator partially assembled data
   struct DiffusionAssemblyData : public AssemblyData
   {
      int dim;
      Tensor<3> oper;
      Tensor<2> dof_quad, dof_quad_d;
      Tensor<2> quad_dof, quad_dof_d;
      Tensor<4> quad_data;
   };
};


} // namespace mfem4

#endif // MFEM4_INTEGRATORS_DIFFUSION_HPP

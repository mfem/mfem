// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PFEM_EXTRAS
#define MFEM_PFEM_EXTRAS

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include <cstddef>

namespace mfem
{

namespace common
{

/** The H1_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an H1_FECollection object.
*/
class H1_ParFESpace : public ParFiniteElementSpace
{
public:
   H1_ParFESpace(ParMesh *m,
                 const int p, const int space_dim = 3,
                 const int type = BasisType::GaussLobatto,
                 int vdim = 1, int order = Ordering::byNODES);
   ~H1_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The ND_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an ND_FECollection object.
*/
class ND_ParFESpace : public ParFiniteElementSpace
{
public:
   ND_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~ND_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The RT_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an RT_FECollection object.
*/
class RT_ParFESpace : public ParFiniteElementSpace
{
public:
   RT_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~RT_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The L2_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an L2_FECollection object.
*/
class L2_ParFESpace : public ParFiniteElementSpace
{
public:
   L2_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~L2_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

class ParDiscreteInterpolationOperator : public ParDiscreteLinearOperator
{
public:
   ParDiscreteInterpolationOperator(ParFiniteElementSpace *dfes,
                                    ParFiniteElementSpace *rfes)
      : ParDiscreteLinearOperator(dfes, rfes) {}
   virtual ~ParDiscreteInterpolationOperator();
};

class ParDiscreteGradOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
                           ParFiniteElementSpace *rfes);
};

class ParDiscreteCurlOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                           ParFiniteElementSpace *rfes);
};

class ParDiscreteDivOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
                          ParFiniteElementSpace *rfes);
};

/// This class computes the irrotational portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class IrrotationalProjector : public Operator
{
public:
   IrrotationalProjector(ParFiniteElementSpace   & H1FESpace,
                         ParFiniteElementSpace   & HCurlFESpace,
                         const int               & irOrder,
                         ParBilinearForm         * s0 = NULL,
                         ParMixedBilinearForm    * weakDiv = NULL,
                         ParDiscreteGradOperator * grad = NULL);
   virtual ~IrrotationalProjector();

   // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the irrotational portion, 'y', of
   // this vector field.  The resulting GridFunction will satisfy Curl y = 0
   // to machine precision.
   virtual void Mult(const Vector &x, Vector &y) const;

   void Update();

private:
   void InitSolver() const;

   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * HCurlFESpace_;

   ParBilinearForm         * s0_;
   ParMixedBilinearForm    * weakDiv_;
   ParDiscreteGradOperator * grad_;

   ParGridFunction * psi_;
   ParGridFunction * xDiv_;

   HypreParMatrix * S0_;
   mutable Vector Psi_;
   mutable Vector RHS_;

   mutable HypreBoomerAMG * amg_;
   mutable HyprePCG       * pcg_;

   Array<int> ess_bdr_, ess_bdr_tdofs_;

   bool ownsS0_;
   bool ownsWeakDiv_;
   bool ownsGrad_;
};

/// This class computes the divergence free portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class DivergenceFreeProjector : public IrrotationalProjector
{
public:
   DivergenceFreeProjector(ParFiniteElementSpace   & H1FESpace,
                           ParFiniteElementSpace   & HCurlFESpace,
                           const int               & irOrder,
                           ParBilinearForm         * s0 = NULL,
                           ParMixedBilinearForm    * weakDiv = NULL,
                           ParDiscreteGradOperator * grad = NULL);
   virtual ~DivergenceFreeProjector();

   // Given a vector 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the divergence free portion, 'y', of
   // this vector field.  The resulting vector will satisfy Div y = 0
   // in a weak sense.
   virtual void Mult(const Vector &x, Vector &y) const;

   void Update();
};


/// Visualize the given parallel mesh object, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   ParMesh &pmesh, const char *title,
                   int x = 0, int y = 0, int w = 400, int h = 400,
                   const char *keys = NULL);

/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    const char *keys = NULL, bool vec = false);

} // namespace common

} // namespace mfem

#endif // MFEM_USE_MPI
#endif

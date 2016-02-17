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

#ifndef MFEM_PFEM_EXTRAS
#define MFEM_PFEM_EXTRAS

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mfem.hpp"
#include <cstddef>

namespace mfem
{

namespace miniapps
{

/** The H1_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an H1_FECollection object.
*/
class H1_ParFESpace : public ParFiniteElementSpace
{
public:
   H1_ParFESpace(ParMesh *m,
                 const int p, const int space_dim = 3, const int type = 0,
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

class ParDiscreteInterpolationOperator
{
public:
   virtual ~ParDiscreteInterpolationOperator();

   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HypreParVector &x, HypreParVector &y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                  double alpha = 1.0, double beta = 0.0);

   /// Computes y = alpha * A^t * x + beta * y
   HYPRE_Int MultTranspose(HypreParVector &x, HypreParVector &y,
                           double alpha = 1.0, double beta = 0.0);

   /// Computes y = alpha * A * x + beta * y
   void Mult(double a, const Vector &x, double b, Vector &y) const;
   /// Computes y = alpha * A^t * x + beta * y
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   /// Computes y = A * x
   void Mult(const Vector &x, Vector &y) const;
   /// Computes y = A^t * x
   void MultTranspose(const Vector &x, Vector &y) const;

   void Update();

   const HypreParMatrix & GetMatrix() const { return *mat_; }

   HypreParMatrix * ParallelAssemble();

protected:
   ParDiscreteInterpolationOperator() : pdlo_(NULL), mat_(NULL) {}

   void createMatrix() const;

   ParDiscreteLinearOperator *pdlo_;
   mutable HypreParMatrix    *mat_;
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
   IrrotationalProjector(ParFiniteElementSpace & H1FESpace,
                         ParFiniteElementSpace & HCurlFESpace,
                         ParDiscreteInterpolationOperator & Grad);
   virtual ~IrrotationalProjector();

   // Given a vector 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the irrotational portion, 'y', of
   // this vector field.  The resulting vector will satisfy Curl y = 0
   // to machine precision.
   virtual void Mult(const Vector &x, Vector &y) const;

   void Update();

private:
   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * HCurlFESpace_;

   ParBilinearForm * s0_;
   ParBilinearForm * m1_;

   HypreBoomerAMG * amg_;
   HyprePCG       * pcg_;
   HypreParMatrix * S0_;
   HypreParMatrix * M1_;
   ParDiscreteInterpolationOperator * Grad_;
   HypreParVector * gradYPot_;
   HypreParVector * yPot_;
   HypreParVector * xDiv_;

   // Array<int> dof_list_;
   Array<int> ess_bdr_;
};

/// This class computes the divergence free portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class DivergenceFreeProjector : public IrrotationalProjector
{
public:
   DivergenceFreeProjector(ParFiniteElementSpace & H1FESpace,
                           ParFiniteElementSpace & HCurlFESpace,
                           ParDiscreteInterpolationOperator & Grad);
   virtual ~DivergenceFreeProjector();

   // Given a vector 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the divergence free portion, 'y', of
   // this vector field.  The resulting vector will satisfy Div y = 0
   // in a weak sense.
   virtual void Mult(const Vector &x, Vector &y) const;

   void Update();

private:
   ParFiniteElementSpace * HCurlFESpace_;
   HypreParVector        * xIrr_;
};


/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400);

} // namespace miniapps

} // namespace mfem

#endif // MFEM_USE_MPI
#endif

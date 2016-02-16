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

#ifndef MFEM_PGRIDFUNC
#define MFEM_PGRIDFUNC

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pfespace.hpp"
#include "gridfunc.hpp"
#include <iostream>
#include <limits>

namespace mfem
{

/// Compute a global Lp norm from the local Lp norms computed by each processor
double GlobalLpNorm(const double p, double loc_norm, MPI_Comm comm);

/// Class for parallel grid function
class ParGridFunction : public GridFunction
{
protected:
   ParFiniteElementSpace *pfes;

   Vector face_nbr_data;

public:
   ParGridFunction() { pfes = NULL; }

   ParGridFunction(ParFiniteElementSpace *pf) : GridFunction(pf), pfes(pf) { }

   /** Construct a ParGridFunction corresponding to *pf and the data from *gf
       which is a local GridFunction on each processor. */
   ParGridFunction(ParFiniteElementSpace *pf, GridFunction *gf);

   /** Creates grid function on (all) dofs from a given vector on the true dofs,
       i.e. P tv. */
   ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv);

   /** Construct a ParGridFunction from the given serial GridFunction.
       If partitioning == NULL (default), the data from 'gf' is NOT copied. */
   ParGridFunction(ParMesh *pmesh, GridFunction *gf, int * partitioning = NULL);

   ParGridFunction &operator=(double value)
   { GridFunction::operator=(value); return *this; }

   ParGridFunction &operator=(const Vector &v)
   { GridFunction::operator=(v); return *this; }

   ParFiniteElementSpace *ParFESpace() { return pfes; }

   void Update() { Update(pfes); }

   void Update(ParFiniteElementSpace *f);

   void Update(ParFiniteElementSpace *f, Vector &v, int v_offset);

   /** Set the grid function on (all) dofs from a given vector on the
       true dofs, i.e. P tv. */
   void Distribute(const Vector *tv);
   void Distribute(const Vector &tv) { Distribute(&tv); }
   void AddDistribute(double a, const Vector *tv);
   void AddDistribute(double a, const Vector &tv)
   { AddDistribute(a, &tv); }

   /// Short semantic for Distribute
   ParGridFunction &operator=(const HypreParVector &tv)
   { Distribute(&tv); return (*this); }

   /// Returns the true dofs in a Vector
   void GetTrueDofs(Vector &tv) const;

   /// Returns the true dofs in a new HypreParVector
   HypreParVector *GetTrueDofs() const;

   /// Returns the vector averaged on the true dofs.
   void ParallelAverage(Vector &tv) const;

   /// Returns the vector averaged on the true dofs.
   void ParallelAverage(HypreParVector &tv) const;

   /// Returns a new vector averaged on the true dofs.
   HypreParVector *ParallelAverage() const;

   /// Returns the vector restricted to the true dofs.
   void ParallelProject(Vector &tv) const;

   /// Returns the vector restricted to the true dofs.
   void ParallelProject(HypreParVector &tv) const;

   /// Returns a new vector restricted to the true dofs.
   HypreParVector *ParallelProject() const;

   /// Returns the vector assembled on the true dofs.
   void ParallelAssemble(Vector &tv) const;

   /// Returns the vector assembled on the true dofs.
   void ParallelAssemble(HypreParVector &tv) const;

   /// Returns a new vector assembled on the true dofs.
   HypreParVector *ParallelAssemble() const;

   void ExchangeFaceNbrData();
   Vector &FaceNbrData() { return face_nbr_data; }
   const Vector &FaceNbrData() const { return face_nbr_data; }

   // Redefine to handle the case when i is a face-neighbor element
   virtual double GetValue(int i, const IntegrationPoint &ip,
                           int vdim = 1) const;
   double GetValue(ElementTransformation &T)
   { return GetValue(T.ElementNo, T.GetIntPoint()); }

   using GridFunction::ProjectCoefficient;
   void ProjectCoefficient(Coefficient &coeff);

   using GridFunction::ProjectDiscCoefficient;
   /** Project a discontinuous vector coefficient as a grid function on a
       continuous parallel finite element space. The values in shared dofs are
       determined from the element with maximal attribute. */
   void ProjectDiscCoefficient(VectorCoefficient &coeff);

   double ComputeL1Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(1.0, GridFunction::ComputeW11Error(
                             *exsol, NULL, 1, NULL, irs), pfes->GetComm());
   }

   double ComputeL1Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, irs); }

   double ComputeL1Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, NULL, irs); }

   double ComputeL2Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeL2Error(exsol, irs),
                          pfes->GetComm());
   }

   double ComputeL2Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(2.0, exsol, NULL, irs); }

   double ComputeL2Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL,
                         Array<int> *elems = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeL2Error(exsol, irs, elems),
                          pfes->GetComm());
   }

   double ComputeMaxError(Coefficient *exsol[],
                          const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(std::numeric_limits<double>::infinity(),
                          GridFunction::ComputeMaxError(exsol, irs),
                          pfes->GetComm());
   }

   double ComputeMaxError(Coefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(std::numeric_limits<double>::infinity(),
                            exsol, NULL, irs);
   }

   double ComputeMaxError(VectorCoefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(std::numeric_limits<double>::infinity(),
                            exsol, NULL, NULL, irs);
   }

   double ComputeLpError(const double p, Coefficient &exsol,
                         Coefficient *weight = NULL,
                         const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(p, GridFunction::ComputeLpError(
                             p, exsol, weight, irs), pfes->GetComm());
   }

   /** When given a vector weight, compute the pointwise (scalar) error as the
       dot product of the vector error with the vector weight. Otherwise, the
       scalar error is the l_2 norm of the vector error. */
   double ComputeLpError(const double p, VectorCoefficient &exsol,
                         Coefficient *weight = NULL,
                         VectorCoefficient *v_weight = NULL,
                         const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(p, GridFunction::ComputeLpError(
                             p, exsol, weight, v_weight, irs), pfes->GetComm());
   }

   virtual void ComputeFlux(BilinearFormIntegrator &blfi,
                            GridFunction &flux,
                            int wcoef = 1, int subdomain = -1);

   /** Save the local portion of the ParGridFunction. It differs from the
       serial GridFunction::Save in that it takes into account the signs of
       the local dofs. */
   virtual void Save(std::ostream &out) const;

   /// Merge the local grid functions
   void SaveAsOne(std::ostream &out = std::cout);

   virtual ~ParGridFunction() { }
};


/** Performs a global L2 projection (through a HypreBoomerAMG solve) of flux
    from supplied discontinuous space into supplied smooth (continuous, or at
    least conforming) space, and computes the Lp norms of the differences
    between them on each element. This is one approach to handling conforming
    and non-conforming elements in parallel. */
void L2ZZErrorEstimator(BilinearFormIntegrator &flux_integrator,
                        ParGridFunction &x,
                        ParFiniteElementSpace &smooth_flux_fes,
                        ParFiniteElementSpace &flux_fes,
                        Vector &errors, int norm_p = 2, double solver_tol = 1e-12,
                        int solver_max_it = 200);

}

#endif // MFEM_USE_MPI

#endif

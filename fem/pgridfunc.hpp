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

   /// Construct a ParGridFunction using previously allocated array @a data.
   /** The ParGridFunction does not assume ownership of @a data which is assumed
       to be of size at least `pf->GetVSize()`. Similar to the GridFunction and
       Vector constructors for externally allocated array, the pointer @a data
       can be NULL. The data array can be replaced later using the method
       SetData().
    */
   ParGridFunction(ParFiniteElementSpace *pf, double *data) :
      GridFunction(pf, data), pfes(pf) { }

   /// Construct a ParGridFunction using a GridFunction as external data.
   /** The parallel space @a *pf and the space used by @a *gf should match. The
       data from @a *gf is used as the local data of the ParGridFunction on each
       processor. The ParGridFunction does not assume ownership of the data. */
   ParGridFunction(ParFiniteElementSpace *pf, GridFunction *gf);

   /** Creates grid function on (all) dofs from a given vector on the true dofs,
       i.e. P tv. */
   ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv);

   /** Construct a ParGridFunction from the given serial GridFunction.
       If partitioning == NULL (default), the data from 'gf' is NOT copied. */
   ParGridFunction(ParMesh *pmesh, GridFunction *gf, int * partitioning = NULL);

   /// Assign constant values to the ParGridFunction data.
   ParGridFunction &operator=(double value)
   { GridFunction::operator=(value); return *this; }

   /// Copy the data from a Vector to the ParGridFunction data.
   ParGridFunction &operator=(const Vector &v)
   { GridFunction::operator=(v); return *this; }

   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   void Update();

   void SetSpace(ParFiniteElementSpace *f);

   /** @brief Make the ParGridFunction reference external data on a new
       ParFiniteElementSpace. */
   /** This method changes the ParFiniteElementSpace associated with the
       ParGridFunction and sets the pointer @a v as external data in the
       ParGridFunction. */
   void MakeRef(ParFiniteElementSpace *f, double *v);

   /** @brief Make the ParGridFunction reference external data on a new
       ParFiniteElementSpace. */
   /** This method changes the ParFiniteElementSpace associated with the
       ParGridFunction and sets the data of the Vector @a v (plus the
       @a v_offset) as external data in the ParGridFunction.
       @note This version of the method will also perform bounds checks when
       the build option MFEM_DEBUG is enabled. */
   void MakeRef(ParFiniteElementSpace *f, Vector &v, int v_offset);

   /** Set the grid function on (all) dofs from a given vector on the
       true dofs, i.e. P tv. */
   void Distribute(const Vector *tv);
   void Distribute(const Vector &tv) { Distribute(&tv); }
   void AddDistribute(double a, const Vector *tv);
   void AddDistribute(double a, const Vector &tv) { AddDistribute(a, &tv); }

   /// Set the GridFunction from the given true-dof vector.
   virtual void SetFromTrueDofs(const Vector &tv) { Distribute(tv); }

   /// Short semantic for Distribute
   ParGridFunction &operator=(const HypreParVector &tv)
   { Distribute(&tv); return (*this); }

   using GridFunction::GetTrueDofs;

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
    and non-conforming elements in parallel. Returns the total error estimate. */
double L2ZZErrorEstimator(BilinearFormIntegrator &flux_integrator,
                          const ParGridFunction &x,
                          ParFiniteElementSpace &smooth_flux_fes,
                          ParFiniteElementSpace &flux_fes,
                          Vector &errors, int norm_p = 2, double solver_tol = 1e-12,
                          int solver_max_it = 200);

}

#endif // MFEM_USE_MPI

#endif

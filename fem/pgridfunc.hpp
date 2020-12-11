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

#ifndef MFEM_PGRIDFUNC
#define MFEM_PGRIDFUNC

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../general/globals.hpp"
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
   ParFiniteElementSpace *pfes; ///< Points to the same object as #fes

   /** @brief Vector used to store data from face-neighbor processors,
       initialized by ExchangeFaceNbrData(). */
   Vector face_nbr_data;

   /** @brief Vector used as an MPI buffer to send face-neighbor data
       in ExchangeFaceNbrData() to neighboring processors. */
   //TODO: Use temporary memory to avoid CUDA malloc allocation cost.
   Vector send_data;

   void ProjectBdrCoefficient(Coefficient *coeff[], VectorCoefficient *vcoeff,
                              Array<int> &attr);

public:
   ParGridFunction() { pfes = NULL; }

   /// Copy constructor. The internal vector #face_nbr_data is not copied.
   ParGridFunction(const ParGridFunction &orig)
      : GridFunction(orig), pfes(orig.pfes) { }

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

   /** @brief Creates grid function on (all) dofs from a given vector on the
       true dofs, i.e. P tv. */
   ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv);

   /** @brief Construct a local ParGridFunction from the given *global*
       GridFunction. If @a partitioning is NULL (default), the data from @a gf
       is NOT copied. */
   ParGridFunction(ParMesh *pmesh, const GridFunction *gf,
                   const int *partitioning = NULL);

   /** @brief Construct a ParGridFunction on a given ParMesh, @a pmesh, reading
       from an std::istream.

       In the process, a ParFiniteElementSpace and a FiniteElementCollection are
       constructed. The new ParGridFunction assumes ownership of both. */
   ParGridFunction(ParMesh *pmesh, std::istream &input);

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** It is assumed that this object and @a rhs use ParFiniteElementSpace%s
       that have the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   ParGridFunction &operator=(const ParGridFunction &rhs)
   { return operator=((const Vector &)rhs); }

   /// Assign constant values to the ParGridFunction data.
   ParGridFunction &operator=(double value)
   { GridFunction::operator=(value); return *this; }

   /// Copy the data from a Vector to the ParGridFunction data.
   ParGridFunction &operator=(const Vector &v)
   { GridFunction::operator=(v); return *this; }

   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   virtual void Update();

   /// Associate a new FiniteElementSpace with the ParGridFunction.
   /** The ParGridFunction is resized using the SetSize() method. The new space
       @a f is expected to be a ParFiniteElementSpace. */
   virtual void SetSpace(FiniteElementSpace *f);

   /// Associate a new parallel space with the ParGridFunction.
   void SetSpace(ParFiniteElementSpace *f);

   using GridFunction::MakeRef;

   /** @brief Make the ParGridFunction reference external data on a new
       FiniteElementSpace. */
   /** This method changes the FiniteElementSpace associated with the
       ParGridFunction and sets the pointer @a v as external data in the
       ParGridFunction. The new space @a f is expected to be a
       ParFiniteElementSpace. */
   virtual void MakeRef(FiniteElementSpace *f, double *v);

   /** @brief Make the ParGridFunction reference external data on a new
       ParFiniteElementSpace. */
   /** This method changes the ParFiniteElementSpace associated with the
       ParGridFunction and sets the pointer @a v as external data in the
       ParGridFunction. */
   void MakeRef(ParFiniteElementSpace *f, double *v);

   /** @brief Make the ParGridFunction reference external data on a new
       FiniteElementSpace. */
   /** This method changes the FiniteElementSpace associated with the
       ParGridFunction and sets the data of the Vector @a v (plus the @a
       v_offset) as external data in the ParGridFunction. The new space @a f is
       expected to be a ParFiniteElementSpace.
       @note This version of the method will also perform bounds checks when
       the build option MFEM_DEBUG is enabled. */
   virtual void MakeRef(FiniteElementSpace *f, Vector &v, int v_offset);

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

   /// Short semantic for Distribute()
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

   // Redefine to handle the case when T describes a face-neighbor element
   virtual double GetValue(ElementTransformation &T, const IntegrationPoint &ip,
                           int comp = 0, Vector *tr = NULL) const;

   virtual void GetVectorValue(int i, const IntegrationPoint &ip,
                               Vector &val) const;

   // Redefine to handle the case when T describes a face-neighbor element
   virtual void GetVectorValue(ElementTransformation &T,
                               const IntegrationPoint &ip,
                               Vector &val, Vector *tr = NULL) const;

   using GridFunction::ProjectCoefficient;
   virtual void ProjectCoefficient(Coefficient &coeff);

   using GridFunction::ProjectDiscCoefficient;
   /** @brief Project a discontinuous vector coefficient as a grid function on
       a continuous finite element space. The values in shared dofs are
       determined from the element with maximal attribute. */
   virtual void ProjectDiscCoefficient(VectorCoefficient &coeff);

   virtual void ProjectDiscCoefficient(Coefficient &coeff, AvgType type);

   virtual void ProjectDiscCoefficient(VectorCoefficient &vcoeff, AvgType type);

   using GridFunction::ProjectBdrCoefficient;

   // Only the values in the master are guaranteed to be correct!
   virtual void ProjectBdrCoefficient(VectorCoefficient &vcoeff,
                                      Array<int> &attr)
   { ProjectBdrCoefficient(NULL, &vcoeff, attr); }

   // Only the values in the master are guaranteed to be correct!
   virtual void ProjectBdrCoefficient(Coefficient *coeff[], Array<int> &attr)
   { ProjectBdrCoefficient(coeff, NULL, attr); }

   // Only the values in the master are guaranteed to be correct!
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                             Array<int> &bdr_attr);

   virtual double ComputeL1Error(Coefficient *exsol[],
                                 const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(1.0, GridFunction::ComputeW11Error(
                             *exsol, NULL, 1, NULL, irs), pfes->GetComm());
   }

   virtual double ComputeL1Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, irs); }

   virtual double ComputeL1Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, NULL, irs); }

   virtual double ComputeL2Error(Coefficient *exsol[],
                                 const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeL2Error(exsol, irs),
                          pfes->GetComm());
   }

   virtual double ComputeL2Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(2.0, exsol, NULL, irs); }

   virtual double ComputeL2Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeL2Error(exsol, irs, elems),
                          pfes->GetComm());
   }

   /// Returns ||grad u_ex - grad u_h||_L2 for H1 or L2 elements
   virtual double ComputeGradError(VectorCoefficient *exgrad,
                                   const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeGradError(exgrad,irs),
                          pfes->GetComm());
   }

   /// Returns ||curl u_ex - curl u_h||_L2 for ND elements
   virtual double ComputeCurlError(VectorCoefficient *excurl,
                                   const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeCurlError(excurl,irs),
                          pfes->GetComm());
   }

   /// Returns ||div u_ex - div u_h||_L2 for RT elements
   virtual double ComputeDivError(Coefficient *exdiv,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeDivError(exdiv,irs),
                          pfes->GetComm());
   }

   /// Returns the Face Jumps error for L2 elements
   virtual double ComputeDGFaceJumpError(Coefficient *exsol,
                                         Coefficient *ell_coeff,
                                         JumpScaling jump_scaling,
                                         const IntegrationRule *irs[]=NULL)
   const;

   /// Returns either the H1-seminorm or the DG Face Jumps error or both
   /// depending on norm_type = 1, 2, 3
   virtual double ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                 Coefficient *ell_coef, double Nu,
                                 int norm_type) const
   {
      return GlobalLpNorm(2.0,
                          GridFunction::ComputeH1Error(exsol,exgrad,ell_coef,
                                                       Nu, norm_type),
                          pfes->GetComm());
   }

   /// Returns the error measured in H1-norm for H1 elements or in "broken"
   /// H1-norm for L2 elements
   virtual double ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                 const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeH1Error(exsol,exgrad,irs),
                          pfes->GetComm());
   }

   /// Returns the error measured H(div)-norm for RT elements
   virtual double ComputeHDivError(VectorCoefficient *exsol,
                                   Coefficient *exdiv,
                                   const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0, GridFunction::ComputeHDivError(exsol,exdiv,irs),
                          pfes->GetComm());
   }

   /// Returns the error measured H(curl)-norm for ND elements
   virtual double ComputeHCurlError(VectorCoefficient *exsol,
                                    VectorCoefficient *excurl,
                                    const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(2.0,
                          GridFunction::ComputeHCurlError(exsol,excurl,irs),
                          pfes->GetComm());
   }

   virtual double ComputeMaxError(Coefficient *exsol[],
                                  const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(infinity(),
                          GridFunction::ComputeMaxError(exsol, irs),
                          pfes->GetComm());
   }

   virtual double ComputeMaxError(Coefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, irs);
   }

   virtual double ComputeMaxError(VectorCoefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, NULL, irs);
   }

   virtual double ComputeLpError(const double p, Coefficient &exsol,
                                 Coefficient *weight = NULL,
                                 const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(p, GridFunction::ComputeLpError(
                             p, exsol, weight, irs), pfes->GetComm());
   }

   /** When given a vector weight, compute the pointwise (scalar) error as the
       dot product of the vector error with the vector weight. Otherwise, the
       scalar error is the l_2 norm of the vector error. */
   virtual double ComputeLpError(const double p, VectorCoefficient &exsol,
                                 Coefficient *weight = NULL,
                                 VectorCoefficient *v_weight = NULL,
                                 const IntegrationRule *irs[] = NULL) const
   {
      return GlobalLpNorm(p, GridFunction::ComputeLpError(
                             p, exsol, weight, v_weight, irs), pfes->GetComm());
   }

   virtual void ComputeFlux(BilinearFormIntegrator &blfi,
                            GridFunction &flux,
                            bool wcoef = true, int subdomain = -1);

   /** Save the local portion of the ParGridFunction. This differs from the
       serial GridFunction::Save in that it takes into account the signs of
       the local dofs. */
   virtual void Save(std::ostream &out) const;

#ifdef MFEM_USE_ADIOS2
   /** Save the local portion of the ParGridFunction. This differs from the
       serial GridFunction::Save in that it takes into account the signs of
       the local dofs. */
   virtual void Save(
      adios2stream &out, const std::string &variable_name,
      const adios2stream::data_type type = adios2stream::data_type::point_data) const;
#endif

   /// Merge the local grid functions
   void SaveAsOne(std::ostream &out = mfem::out);

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

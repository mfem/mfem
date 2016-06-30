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

#ifndef MFEM_GRIDFUNC
#define MFEM_GRIDFUNC

#include "../config/config.hpp"
#include "fespace.hpp"
#include "coefficient.hpp"
#include "bilininteg.hpp"
#include <limits>
#include <ostream>
#include <string>

namespace mfem
{

/// Class for grid function - Vector with associated FE space.
class GridFunction : public Vector
{
protected:
   /// FE space on which grid function lives.
   FiniteElementSpace *fes;

   /// Used when the grid function is read from a file
   FiniteElementCollection *fec;

   long sequence; // see FiniteElementSpace::sequence, Mesh::sequence

   void SaveSTLTri(std::ostream &out, double p1[], double p2[], double p3[]);

   void GetVectorGradientHat(ElementTransformation &T, DenseMatrix &gh);

   // Project the delta coefficient without scaling and return the (local)
   // integral of the projection.
   void ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                double &integral);

   // Sum fluxes to vertices and count element contributions
   void SumFluxAndCount(BilinearFormIntegrator &blfi,
                        GridFunction &flux,
                        Array<int>& counts,
                        int wcoef,
                        int subdomain);

   /** Project a discontinuous vector coefficient in a continuous space and
       return in dof_attr the maximal attribute of the elements containing each
       degree of freedom. */
   void ProjectDiscCoefficient(VectorCoefficient &coeff, Array<int> &dof_attr);

   void Destroy();

public:

   GridFunction() { fes = NULL; fec = NULL; sequence = 0; }

   /// Creates grid function associated with *f.
   GridFunction(FiniteElementSpace *f) : Vector(f->GetVSize())
   { fes = f; fec = NULL; sequence = f->GetSequence(); }

   GridFunction(Mesh *m, std::istream &input);

   GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces);

   /// Make the GridFunction the owner of 'fec' and 'fes'
   void MakeOwner(FiniteElementCollection *_fec) { fec = _fec; }

   FiniteElementCollection *OwnFEC() { return fec; }

   int VectorDim() const;

   /// Returns the values in the vertices of i'th element for dimension vdim.
   void GetNodalValues(int i, Array<double> &nval, int vdim = 1) const;

   virtual double GetValue(int i, const IntegrationPoint &ip,
                           int vdim = 1) const;

   void GetVectorValue(int i, const IntegrationPoint &ip, Vector &val) const;

   void GetValues(int i, const IntegrationRule &ir, Vector &vals,
                  int vdim = 1) const;

   void GetValues(int i, const IntegrationRule &ir, Vector &vals,
                  DenseMatrix &tr, int vdim = 1) const;

   int GetFaceValues(int i, int side, const IntegrationRule &ir, Vector &vals,
                     DenseMatrix &tr, int vdim = 1) const;

   void GetVectorValues(ElementTransformation &T, const IntegrationRule &ir,
                        DenseMatrix &vals) const;

   void GetVectorValues(int i, const IntegrationRule &ir,
                        DenseMatrix &vals, DenseMatrix &tr) const;

   int GetFaceVectorValues(int i, int side, const IntegrationRule &ir,
                           DenseMatrix &vals, DenseMatrix &tr) const;

   void GetValuesFrom(GridFunction &);

   void GetBdrValuesFrom(GridFunction &);

   void GetVectorFieldValues(int i, const IntegrationRule &ir,
                             DenseMatrix &vals,
                             DenseMatrix &tr, int comp = 0) const;

   /// For a vector grid function, makes sure that the ordering is byNODES.
   void ReorderByNodes();

   /// Return the values as a vector on mesh vertices for dimension vdim.
   void GetNodalValues(Vector &nval, int vdim = 1) const;

   void GetVectorFieldNodalValues(Vector &val, int comp) const;

   void ProjectVectorFieldOn(GridFunction &vec_field, int comp = 0);

   void GetDerivative(int comp, int der_comp, GridFunction &der);

   double GetDivergence(ElementTransformation &tr);

   void GetCurl(ElementTransformation &tr, Vector &curl);

   void GetGradient(ElementTransformation &tr, Vector &grad);

   void GetGradients(const int elem, const IntegrationRule &ir,
                     DenseMatrix &grad);

   void GetVectorGradient(ElementTransformation &tr, DenseMatrix &grad);

   /** Compute \f$ (\int_{\Omega} (*this) \psi_i)/(\int_{\Omega} \psi_i) \f$,
       where \f$ \psi_i \f$ are the basis functions for the FE space of avgs.
       Both FE spaces should be scalar and on the same mesh. */
   void GetElementAverages(GridFunction &avgs);

   /** Impose the given bounds on the function's DOFs while preserving its local
    *  integral (described in terms of the given weights) on the i'th element
    *  through SLBPQ optimization.
    *  Intended to be used for discontinuous FE functions. */
   void ImposeBounds(int i, const Vector &weights,
                     const Vector &_lo, const Vector &_hi);
   void ImposeBounds(int i, const Vector &weights,
                     double _min = 0.0, double _max = std::numeric_limits<double>::infinity());

   /** Project the given 'src' GridFunction to 'this' GridFunction, both of
       which must be on the same mesh. The current implementation assumes that
       all element use the same projection matrix. */
   void ProjectGridFunction(const GridFunction &src);

   void ProjectCoefficient(Coefficient &coeff);

   // call fes -> BuildDofToArrays() before using this projection
   void ProjectCoefficient(Coefficient &coeff, Array<int> &dofs, int vd = 0);

   void ProjectCoefficient(VectorCoefficient &vcoeff);

   // call fes -> BuildDofToArrays() before using this projection
   void ProjectCoefficient(VectorCoefficient &vcoeff, Array<int> &dofs);

   void ProjectCoefficient(Coefficient *coeff[]);

   /** Project a discontinuous vector coefficient as a grid function on a
       continuous finite element space. The values in shared dofs are determined
       from the element with maximal attribute. */
   void ProjectDiscCoefficient(VectorCoefficient &coeff);

   void ProjectBdrCoefficient(Coefficient &coeff, Array<int> &attr)
   {
      Coefficient *coeff_p = &coeff;
      ProjectBdrCoefficient(&coeff_p, attr);
   }

   void ProjectBdrCoefficient(Coefficient *coeff[], Array<int> &attr);

   /** Project the normal component of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes RT-type VectorFE GridFunction. */
   void ProjectBdrCoefficientNormal(VectorCoefficient &vcoeff,
                                    Array<int> &bdr_attr);

   /** Project the tangential components of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes ND-type VectorFE GridFunction. */
   void ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                     Array<int> &bdr_attr);

   double ComputeL2Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(2.0, exsol, NULL, irs); }

   double ComputeL2Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const;

   double ComputeL2Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL,
                         Array<int> *elems = NULL) const;

   double ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                         Coefficient *ell_coef, double Nu,
                         int norm_type) const;

   double ComputeMaxError(Coefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(std::numeric_limits<double>::infinity(),
                            exsol, NULL, irs);
   }

   double ComputeMaxError(Coefficient *exsol[],
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeMaxError(VectorCoefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(std::numeric_limits<double>::infinity(),
                            exsol, NULL, NULL, irs);
   }

   double ComputeL1Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, irs); }

   double ComputeW11Error(Coefficient *exsol, VectorCoefficient *exgrad,
                          int norm_type, Array<int> *elems = NULL,
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeL1Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, NULL, irs); }

   double ComputeLpError(const double p, Coefficient &exsol,
                         Coefficient *weight = NULL,
                         const IntegrationRule *irs[] = NULL) const;

   /** When given a vector weight, compute the pointwise (scalar) error as the
       dot product of the vector error with the vector weight. Otherwise, the
       scalar error is the l_2 norm of the vector error. */
   double ComputeLpError(const double p, VectorCoefficient &exsol,
                         Coefficient *weight = NULL,
                         VectorCoefficient *v_weight = NULL,
                         const IntegrationRule *irs[] = NULL) const;

   virtual void ComputeFlux(BilinearFormIntegrator &blfi,
                            GridFunction &flux,
                            int wcoef = 1, int subdomain = -1);

   /// Redefine '=' for GridFunction = constant.
   GridFunction &operator=(double value);

   GridFunction &operator=(const Vector &v);

   GridFunction &operator=(const GridFunction &v);

   /// Transform by the Space UpdateMatrix (e.g., on Mesh change).
   void Update();

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   void SetSpace(FiniteElementSpace *f);

   void MakeRef(FiniteElementSpace *f, Vector &v, int v_offset);

   /// Save the GridFunction to an output stream.
   virtual void Save(std::ostream &out) const;

   /** Write the GridFunction in VTK format. Note that Mesh::PrintVTK must be
       called first. The parameter ref > 0 must match the one used in
       Mesh::PrintVTK. */
   void SaveVTK(std::ostream &out, const std::string &field_name, int ref);

   void SaveSTL(std::ostream &out, int TimesToRefine = 1);

   /// Destroys grid function.
   virtual ~GridFunction() { Destroy(); }
};


/** Overload operator<< for std::ostream and GridFunction; valid also for the
    derived class ParGridFunction */
std::ostream &operator<<(std::ostream &out, const GridFunction &sol);


double ZZErrorEstimator(BilinearFormIntegrator &blfi,
                        GridFunction &u,
                        GridFunction &flux,
                        Vector &error_estimates,
                        Array<int> *aniso_flags = NULL,
                        int with_subdomains = 1);

/// Compute the Lp distance between two grid functions on the given element.
double ComputeElementLpDistance(double p, int i,
                                GridFunction& gf1, GridFunction& gf2);


/// Class used for extruding scalar GridFunctions
class ExtrudeCoefficient : public Coefficient
{
private:
   int n;
   Mesh *mesh_in;
   Coefficient &sol_in;
public:
   ExtrudeCoefficient(Mesh *m, Coefficient &s, int _n)
      : n(_n), mesh_in(m), sol_in(s) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ExtrudeCoefficient() { }
};

/// Extrude a scalar 1D GridFunction, after extruding the mesh with Extrude1D.
GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny);

}

#endif
